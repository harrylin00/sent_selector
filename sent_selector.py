import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F

class SentSelector(nn.Module):
    def __init__(self, config, word_embed):
        super().__init__()

        self.config = config
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layer = config['num_layer']
        self.dropout_prob = config['dropout']

        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(word_embed), freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.query_linear = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.para_linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.query_encoder = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layer,
                                     batch_first=True, dropout=self.dropout_prob, bidirectional=True)
        self.para_encoder = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layer,
                                     batch_first=True, dropout=self.dropout_prob, bidirectional=True)

    def train_forward(self, query, query_len, paragraph, paragraph_len, query_to_para_idx):
        query_hidden = self.encode_query(query, query_len)
        paragraph_hidden = self.encode_paragraph(paragraph, paragraph_len)
        similarity = self.compute_similarity(query_hidden, paragraph_hidden)
        return similarity

    def predict_forward(self, query, query_len, paragraph, paragraph_len, query_to_para_idx):
        query_hidden = self.encode_query(query, query_len)
        paragraph_hidden = self.encode_paragraph(paragraph, paragraph_len)
        similarity = self.predict_compute_similarity(query_hidden, paragraph_hidden, query_to_para_idx)
        return similarity

    def encode_query(self, query, query_len):
        batch_size = query.size(0)

        query_embed = self.word_embed(query)
        query_embed = self.dropout(query_embed)
        query_embed = pack_padded_sequence(query_embed, query_len, batch_first=True, enforce_sorted=False)

        out, (h, c) = self.query_encoder(query_embed)    # query_hidden [num_layer * bidirection, batch, hidden_size]
        h = h.view(self.num_layer, -1, batch_size, self.hidden_size).permute(0, 2, 1, 3)[-1].reshape(batch_size, -1)
        query_hidden = torch.tanh(h)
        query_hidden = self.query_linear(query_hidden)
        return query_hidden

    def encode_paragraph(self, paragraph, paragraph_len):
        batch_size = paragraph.size(0)

        para_embed = self.word_embed(paragraph)
        para_embed = self.dropout(para_embed)
        para_embed = pack_padded_sequence(para_embed, paragraph_len, batch_first=True, enforce_sorted=False)

        out, (h, c) = self.para_encoder(para_embed)
        h = h.view(self.num_layer, -1, batch_size, self.hidden_size).permute(0, 2, 1, 3)[-1].reshape(batch_size, -1)
        para_hidden = torch.tanh(h)
        para_hidden = self.para_linear(para_hidden)
        return para_hidden

    def compute_similarity(self, query, paragraph):
        """ compute the similarity between query and related paragraph """
        batch_size = query.size(0)

        query = query.reshape(batch_size, self.hidden_size, 1)
        paragraph = paragraph.reshape(batch_size, -1, self.hidden_size)
        sim_result = torch.bmm(paragraph, query).squeeze()
        return sim_result

    def predict_compute_similarity(self, query, paragraph, query_to_para_idx):
        batch_size = query.size(0)
        similarity = []

        for i in range(batch_size):
            que = query[i].unsqueeze(1).clone()  # [hidden, 1]
            para = paragraph[query_to_para_idx[i] : query_to_para_idx[i + 1]].clone()   #[num, hidden]
            sim = torch.mm(para, que).squeeze()   # [num]
            similarity.append(sim)
        return similarity
