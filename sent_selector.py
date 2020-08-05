import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F

from charCNN import CharCNN

class SentSelector(nn.Module):
    def __init__(self, config, word_embed=None, bert_config=None):
        super().__init__()

        self.config = config
        self.input_size = 300 if config['embed_method']=='glove' else bert_config.hidden_size
        self.hidden_size = config['hidden_size']
        self.num_layer = config['num_layer']
        self.dropout_prob = config['dropout']

        if config['embed_method'] == 'glove':
            self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(word_embed), freeze=False, padding_idx=0)
        elif config['embed_method'] == 'bert':
            self.word_embed = word_embed
            if config['freeze_bert']:
                self.word_embed.weight.requires_grad=False   # for Bert model, we will freeze the word embeddings

        if config['use_charCNN']:
            self.cnn_embed = CharCNN(config)
            self.input_size += config['cnn_hidden_size']

        if config['sample_method'] == 'pair':
            self.diff_linear = nn.Linear(1, 1)

        self.dropout = nn.Dropout(self.dropout_prob)

        self.query_linear = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.para_linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.query_encoder = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layer,
                                     batch_first=True, dropout=self.dropout_prob, bidirectional=True)
        self.para_encoder = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layer,
                                     batch_first=True, dropout=self.dropout_prob, bidirectional=True)


    def train_forward(self, query, query_char, query_len, paragraph, paragraph_char, paragraph_len, query_to_para_idx):
        """
        It is the same with predict_forward
        :param query: [batch, seq_len]
        :param query_char:  [batch, seq_len, word_len, alpha_len + 1]
        :param query_len: [batch, seq_len]
        :param paragraph:
        :param paragraph_char:
        :param paragraph_len:
        :param query_to_para_idx:
        :return:
        """

        query_hidden= self.encode_query(query, query_char, query_len)   # out: [batch, seq_len, hidden]
        paragraph_hidden = self.encode_paragraph(paragraph, paragraph_char, paragraph_len)   #[ir+re * batch, seq_len, hidden]
        similarity = self.compute_similarity(query_hidden, paragraph_hidden)    # [batch, ir+re]
        if self.config['sample_method'] == 'list':
            return similarity
        else:   # pair-wise, following RankNet idea
            prob = self.pair_prob(similarity)
            return prob

    def predict_forward(self, query, query_char, query_len, paragraph, paragraph_char, paragraph_len, query_to_para_idx):
        query_hidden = self.encode_query(query, query_char, query_len)
        paragraph_hidden = self.encode_paragraph(paragraph, paragraph_char, paragraph_len)
        similarity = self.predict_compute_similarity(query_hidden, paragraph_hidden, query_to_para_idx)
        if self.config['sample_method'] == 'pair':
            similarity = [torch.sigmoid(sim) for sim in similarity]
        return similarity

    def encode_query(self, query, query_char, query_len):
        batch_size, seq_len = query.size(0), query.size(1)

        query_embed = self.word_embed(query)
        query_embed = self.dropout(query_embed)

        if self.config['use_charCNN']:
            query_char = query_char.reshape(
                batch_size * seq_len, self.config['cnn_word_len'], self.config['cnn_input_size'])\
                .permute(0, 2, 1)       # [batch * seq_len, input_size, word_len]
            query_char_embed = self.cnn_embed(query_char).reshape(batch_size, seq_len, -1)  # [batch, seq_len, hidden]
            query_embed = torch.cat((query_embed, query_char_embed), dim=-1)

        query_embed = pack_padded_sequence(query_embed, query_len, batch_first=True, enforce_sorted=False)

        out, (h, c) = self.query_encoder(query_embed)    # query_hidden [num_layer * bidirection, batch, hidden_size]
        h = h.view(self.num_layer, -1, batch_size, self.hidden_size).permute(0, 2, 1, 3)[-1].reshape(batch_size, -1)
        query_hidden = torch.tanh(h)
        query_hidden = self.query_linear(query_hidden)
        return query_hidden

    def encode_paragraph(self, paragraph, paragraph_char, paragraph_len):
        batch_size, seq_len = paragraph.size(0), paragraph.size(1)

        para_embed = self.word_embed(paragraph)
        para_embed = self.dropout(para_embed)

        if self.config['use_charCNN']:
            paragraph_char = paragraph_char.reshape(
                batch_size * seq_len, self.config['cnn_word_len'], self.config['cnn_input_size'])\
                .permute(0, 2, 1)       # [batch * seq_len, input_size, word_len]
            paragraph_char_embed = self.cnn_embed(paragraph_char).reshape(batch_size, seq_len, -1)  # [batch, seq_len, hidden]
            para_embed = torch.cat((para_embed, paragraph_char_embed), dim=-1)

        para_embed = pack_padded_sequence(para_embed, paragraph_len, batch_first=True, enforce_sorted=False)

        out, (h, c) = self.para_encoder(para_embed)
        h = h.view(self.num_layer, -1, batch_size, self.hidden_size).permute(0, 2, 1, 3)[-1].reshape(batch_size, -1)
        para_hidden = torch.tanh(h)
        para_hidden = self.para_linear(para_hidden)

        # get last layer's out for attention
        # out, out_lens = pad_packed_sequence(out, batch_first=True)  # [batch, seq_len, hidden * bidirection]
        return para_hidden

    def compute_similarity(self, query, paragraph):
        """ compute the similarity between query and related paragraph """
        batch_size = query.size(0)

        query = query.reshape(batch_size, self.hidden_size, 1)
        paragraph = paragraph.reshape(batch_size, -1, self.hidden_size)
        sim_result = torch.bmm(paragraph, query).squeeze(dim=-1)    # [batch_size, rel+irrel num]
        sim_result = sim_result / torch.sqrt(self.hidden_size)
        return sim_result

    def predict_compute_similarity(self, query, paragraph, query_to_para_idx):
        batch_size = query.size(0)
        similarity = []

        for i in range(batch_size):
            que = query[i].unsqueeze(1).clone()  # [hidden, 1]
            para = paragraph[query_to_para_idx[i] : query_to_para_idx[i + 1]].clone()   #[num, hidden]
            sim = torch.mm(para, que).squeeze()   # [num]
            sim = sim / torch.sqrt(self.hidden_size)
            if self.config['sample_method'] == 'list':
                sim = F.softmax(sim, dim=-1)
            similarity.append(sim)
        return similarity

    def pair_prob(self, similarity):
        """
        :param similarity: [batch, 2], compute the prob that first doc has a higher relevance
        :return:
        """
        similarity = torch.sigmoid(similarity)
        diff = (similarity[:, 0] - similarity[:, 1])   # [batch]
        # diff = torch.sigmoid(diff)
        return diff
