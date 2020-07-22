import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F

class ESIM(nn.Module):
    def __init__(self, config, word_embed=None, bert_config=None):
        super().__init__()
        self.config = config
        self.input_size = 300 if config['embed_method'] == 'glove' else bert_config.hidden_size
        self.hidden_size = config['hidden_size']
        self.num_layer = config['num_layer']
        self.dropout_prob = config['dropout']
        self.linear_size = config['linear_size']

        if config['embed_method'] == 'glove':
            self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(word_embed), freeze=False, padding_idx=0)
        elif config['embed_method'] == 'bert':
            self.word_embed = word_embed
            if config['freeze_bert']:
                self.word_embed.weight.requires_grad=False   # for Bert model, we will freeze the word embeddings

        self.dropout = nn.Dropout(self.dropout_prob)

        self.query_encoder = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layer,
                                     batch_first=True, dropout=self.dropout_prob, bidirectional=True)
        self.para_encoder = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layer,
                                    batch_first=True, dropout=self.dropout_prob, bidirectional=True)

        self.infer_query_encoder = nn.LSTM(self.hidden_size * 8, self.hidden_size,
                                           num_layers=1, batch_first=True, bidirectional=True)
        self.infer_para_encoder = nn.LSTM(self.hidden_size * 8, self.hidden_size,
                                          num_layers=1, batch_first=True, bidirectional=True)
        self.fc = self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.linear_size, 1)
        )

    def train_forward(self,query, query_char, query_len, paragraph, paragraph_char, paragraph_len, query_to_para_idx):
        query_out = self.encode_query(query, query_char, query_len)  # out: [batch, seq_len, hidden]
        paragraph_out = self.encode_paragraph(paragraph, paragraph_char, paragraph_len)  # [ir+re * batch, seq_len, hidden]

        (batch_size, q_seq_len, hidden_size), p_seq_len = query_out.shape, paragraph_out.size(1)
        paragraph_out = paragraph_out.reshape(batch_size, -1, p_seq_len, hidden_size)   #[batch, (ir+re), seq_len, hidden]
        # pq_attention = torch.bmm(query_out, paragraph_out.transpose(1, 2))\
        #     .reshape(batch_size, q_seq_len, p_seq_len, -1)\
        #     .permute(0, 3, 1, 2)  # [batch, (ir+re), q_seq_len, p_seq_len]

        query_mask = query.eq(0)    # [batch, q_seq]
        paragraph_mask = paragraph.eq(0).reshape(batch_size, -1, p_seq_len)   #[batch, (re+ir), p_seq_len]

        similarity = torch.zeros((batch_size, paragraph_out.size(1))).to(self.config['device'])
        for batch_idx in range(batch_size):
            # [ir+re, seq_len, hidden]
            p_out = paragraph_out[batch_idx].reshape(-1, p_seq_len, hidden_size)
            q_out = query_out[batch_idx].repeat(p_out.size(0), 1, 1)
            sub_batch = p_out.size(0)

            # compute attention based vector
            pq_atten = torch.bmm(q_out, p_out.transpose(1, 2))  #[(ir+re), q_seq, p_seq]
            q_mask = query_mask[batch_idx].repeat(p_seq_len, 1)
            p_mask = paragraph_mask[batch_idx].repeat(q_seq_len, 1, 1).permute(1, 0, 2)
            # q_mask = query_mask[batch_idx].expand(p_seq_len, q_seq_len)
            # p_mask = paragraph_mask[batch_idx].expand(q_seq_len, sub_batch, p_seq_len).permute(1, 0, 2)  # [sub_batch, q_seq, p_seq_len]
            p_out_align = torch.bmm(F.softmax(pq_atten.permute(0, 2, 1).masked_fill(q_mask, value=-1000), dim=-1), q_out)
            q_out_align = torch.bmm(F.softmax(pq_atten.masked_fill(p_mask, value=-1000), dim=-1), p_out)

            # concatenate all info: [encoding, attention based, element-wise, difference]
            p_combine = torch.cat([p_out, p_out_align, p_out * p_out_align, p_out - p_out_align], dim=-1)
            q_combine = torch.cat([q_out, q_out_align, q_out * q_out_align, q_out - q_out_align], dim=-1)

            # Infer encoding
            p_len = paragraph_len[batch_idx * sub_batch: (batch_idx + 1) * sub_batch]
            q_len = query_len[batch_idx].expand(sub_batch)
            p_combine_out, q_combine_out = self.infer_encode(p_combine, p_len, q_combine, q_len)

            # Pooling and compute the similarity
            sim = self.pool_and_compute_sim(p_combine_out, q_combine_out)
            similarity[batch_idx] = sim
        return similarity

    def predict_forward(self, query, query_char, query_len, paragraph, paragraph_char, paragraph_len,
                        query_to_para_idx):
        similarity = []
        query_out = self.encode_query(query, query_char, query_len)  # out: [batch, seq_len, hidden]
        paragraph_out = self.encode_paragraph(paragraph, paragraph_char,
                                              paragraph_len)  # [ir+re * batch, seq_len, hidden]

        (batch_size, q_seq_len, hidden_size), p_seq_len = query_out.shape, paragraph_out.size(1)
        for batch_idx in range(batch_size):
            p_out = paragraph_out[query_to_para_idx[batch_idx]: query_to_para_idx[batch_idx + 1]].clone()   # [ire+re, p_seq_len, hidden]
            q_out = query_out[batch_idx].clone().repeat(p_out.size(0), 1, 1)    # [ire+re, q_seq_len, hidden]
            sub_batch = p_out.size(0)

            # compute attention based vector
            pq_atten = torch.bmm(q_out, p_out.transpose(-1, -2))    #[ire+re, q_seq, p_seq]
            q_mask = query[batch_idx].eq(0).repeat(p_seq_len, 1)
            p_mask = paragraph[query_to_para_idx[batch_idx] : query_to_para_idx[batch_idx + 1]].eq(0)\
                .repeat(q_seq_len, 1, 1).permute(1, 0, 2)
            # q_mask = query[batch_idx].eq(0).expand(p_seq_len, q_seq_len)
            # p_mask = paragraph[query_to_para_idx[batch_idx] : query_to_para_idx[batch_idx + 1]].eq(0)\
            #     .expand(q_seq_len, sub_batch, p_seq_len).permute(1, 0, 2)   #[ire+re, q_seq, p_len]
            p_out_align = torch.bmm(
                F.softmax(pq_atten.permute(0, 2, 1).masked_fill(q_mask, value=-1000), dim=-1), q_out)
            q_out_align = torch.bmm(F.softmax(pq_atten.masked_fill(p_mask, value=-1000), dim=-1), p_out)

            # concatenate all info: [encoding, attention based, element-wise, difference]
            p_combine = torch.cat([p_out, p_out_align, p_out * p_out_align, p_out - p_out_align], dim=-1)
            q_combine = torch.cat([q_out, q_out_align, q_out * q_out_align, q_out - q_out_align], dim=-1)

            # Infer encoding
            p_len = paragraph_len[query_to_para_idx[batch_idx]: query_to_para_idx[batch_idx + 1]]
            q_len = query_len[batch_idx].expand(sub_batch)
            p_combine_out, q_combine_out = self.infer_encode(p_combine, p_len, q_combine, q_len)

            # Pooling and compute the similarity
            sim = self.pool_and_compute_sim(p_combine_out, q_combine_out)
            sim = F.softmax(sim, dim=-1)
            similarity.append(sim)
        return similarity

    def encode_query(self, query, query_char, query_len):
        batch_size, seq_len = query.size(0), query.size(1)

        query_embed = self.word_embed(query)
        query_embed = self.dropout(query_embed)
        query_embed = pack_padded_sequence(query_embed, query_len, batch_first=True, enforce_sorted=False)

        out, (h, c) = self.query_encoder(query_embed)    # query_hidden [num_layer * bidirection, batch, hidden_size]
        out, out_lens = pad_packed_sequence(out, batch_first=True)  # [batch, seq_len, hidden * bidirection]
        return out

    def encode_paragraph(self, paragraph, paragraph_char, paragraph_len):
        batch_size, seq_len = paragraph.size(0), paragraph.size(1)

        para_embed = self.word_embed(paragraph)
        para_embed = self.dropout(para_embed)
        para_embed = pack_padded_sequence(para_embed, paragraph_len, batch_first=True, enforce_sorted=False)

        out, (h, c) = self.para_encoder(para_embed)
        # get last layer's out for attention
        out, out_lens = pad_packed_sequence(out, batch_first=True)  # [batch, seq_len, hidden * bidirection]
        return out

    def infer_encode(self, p_combine, p_len, q_combine, q_len):
        p_combine = pack_padded_sequence(p_combine, p_len,
                                         batch_first=True, enforce_sorted=False)
        q_combine = pack_padded_sequence(q_combine, q_len,
                                         batch_first=True, enforce_sorted=False)
        p_combine_out, _ = self.infer_para_encoder(p_combine)
        q_combine_out, _ = self.infer_query_encoder(q_combine)
        p_combine_out, _ = pad_packed_sequence(p_combine_out, batch_first=True)
        q_combine_out, _ = pad_packed_sequence(q_combine_out, batch_first=True)
        return p_combine_out, q_combine_out

    def pool_and_compute_sim(self, p_combine_out, q_combine_out):
        p_combine_out = self.apply_multiple(p_combine_out)
        q_combine_out = self.apply_multiple(q_combine_out)
        x = torch.cat([p_combine_out, q_combine_out], dim=-1)
        sim = self.fc(x).squeeze()
        return sim

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2).contiguous(), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2).contiguous(), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)