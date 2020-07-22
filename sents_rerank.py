from transformers import BertTokenizer, BertModel, BertConfig
import os
import torch
from typing import List
from torch.nn.utils.rnn import *

from esim import ESIM
from sent_selector import SentSelector

class Sent_rerank():
    def __init__(self, config):
        self.config = config
        self.model, self.word2idx, self.bert_tokenizer = self._get_model()
        self.model.eval()
        self.model.to(self.config['device'])

    def rerank(self, query: str, sents: List[str]):
        """
        Get query string and list of sentences as a input, rerank them by model's score
        Return list of sents and the corresponding scores in descending.
        Can add topk function to filter less but higher quality sentences
        """
        query_tensor, query_char_tensor, query_len, paragraph_tensor, paragraph_char_tensor, paragraph_len = \
            self._get_query_para_tensor(self.config, query, sents, word2idx=self.word2idx, bert_tokenizer=self.bert_tokenizer)

        query_to_sents_idx = [0, len(sents)]
        sim = self.model.predict_forward(query_tensor, query_char_tensor, query_len,
                                    paragraph_tensor, paragraph_char_tensor, paragraph_len,
                                    query_to_sents_idx)  # List[(para_num)]

        if self.config['use_lexical']:  # use lexical information by computing tf-idf values
            lexical_sim = self._compute_lexical_similarity(self.config, query, sents, query_to_sents_idx)
            alpha = self.config['lexical_alpha']
            sim = [alpha * s + (1 - alpha) * ls for s, ls in zip(sim, lexical_sim)]

        # Since we only have one query, so we directly choose the 0-idx's tensor
        sim = sim[0]
        rerank_sents, rerank_sim = self._rerank_sents_by_sim(sents, sim)

        #TODO: Add filtering function to filter less but higher quality sentences
        return rerank_sents, rerank_sim

    def _get_model(self):
        if self.config['embed_method'] == 'bert':
            # bert info
            tokenizer = BertTokenizer.from_pretrained(self.config['bert_config'])
            bert_config = BertConfig().from_pretrained(self.config['bert_config'])
            word_embed = BertModel.from_pretrained(self.config['bert_config']).get_input_embeddings()
            word2idx = None
        elif self.config['embed_method'] == 'glove':
            word2idx, word_embed = self._read_glove(self.config['glove_path'])
            tokenizer = None

        # model setting
        if self.config['use_esim']:
            model = ESIM(self.config, word_embed=word_embed, bert_config=bert_config)
        else:
            model = SentSelector(self.config, word_embed=word_embed, bert_config=bert_config)

        if os.path.exists(self.config['model_load_path']) and self.config['is_load_model']:
            print('begin to load model:', self.config['model_load_path'])
            model.load_state_dict(torch.load(self.config['model_load_path'], map_location=self.config['device']))
        return model, word2idx, tokenizer

    def _read_glove(self, filepath):
        """
        get word2idx and word_embed
        NOTE: set <PAD> as word_idx = 0, embed_size is 300d
        """
        print('reading glove files:', filepath)

        word2idx = {}
        word_embed = [[0] * 300]  # word_embed[0] = [0] * 300, represent the <PAD>

        with open(filepath, 'r') as f:
            for idx, line in enumerate(f):
                line_list = line.split()
                word = ' '.join(line_list[: len(line_list) - 300])
                embed = [float(num) for num in line_list[len(line_list) - 300:]]

                word2idx[word] = idx + 1
                word_embed.append(embed)

        return word2idx, word_embed

    def _get_query_para_tensor(self, config, query, paragraphs, word2idx=None, bert_tokenizer=None):
        """ encode query and paragraphs into vector with tensor version"""
        if config['embed_method'] == 'bert':
            # tokenize the query and paragraphs
            query = self._bert_tokenize(query, bert_tokenizer)
            paragraphs = self._bert_tokenize(paragraphs, bert_tokenizer)

            # vectorize, simply choose [1:-1] because of removing [CLS] and [SEP]
            # bert is restricted with max_len=512
            query_tensor = [bert_tokenizer.encode(que, max_length=512, truncation=True)[1:-1] for que in query]
            paragraphs_tensor = [bert_tokenizer.encode(para, max_length=512, truncation=True)[1:-1] for para in
                                 paragraphs]
        elif config['embed_method'] == 'glove':
            query_tensor = self._vectorize_to_tensor(query, word2idx=word2idx)
            paragraphs_tensor = self._vectorize_to_tensor(paragraphs, word2idx=word2idx)

        if config['use_charCNN']:
            query_char_tensor = self._vectorize_to_char_tensor(config, query, alpha_dict=config['alpha_dict'])
            paragraph_char_tensor = self._vectorize_to_char_tensor(config, paragraphs, alpha_dict=config['alpha_dict'])

            query_char_tensor = pad_sequence(query_char_tensor, batch_first=True)
            paragraph_char_tensor = pad_sequence(paragraph_char_tensor, batch_first=True)

        query_tensor = [torch.LongTensor(q) for q in query_tensor]
        query_len = torch.LongTensor([len(q) for q in query_tensor])
        query_pad = pad_sequence(query_tensor, batch_first=True)

        paragraphs_tensor = [torch.LongTensor(para) for para in paragraphs_tensor]
        paragraphs_len = torch.LongTensor([len(para) for para in paragraphs_tensor])
        paragraphs_pad = pad_sequence(paragraphs_tensor, batch_first=True)

        device = config['device']
        if config['use_charCNN']:
            return query_pad.to(device), query_char_tensor.to(device), query_len.to(device), \
                   paragraphs_pad.to(device), paragraph_char_tensor.to(device), paragraphs_len.to(device)
        else:
            return query_pad.to(device), None, query_len.to(device), \
                   paragraphs_pad.to(device), None, paragraphs_len.to(device)

    def _bert_tokenize(self, list_strs, bert_tokenizer):
        res = []
        for strs in list_strs:
            tokenized_res = []
            for token in strs:
                tokenized_res.extend(bert_tokenizer.tokenize(token))
            if len(
                    tokenized_res) == 0:  # remove the case when tokenized_res = [], else it will raise an error when encoding
                tokenized_res = ['']
            res.append(tokenized_res)
        return res

    def _vectorize_to_tensor(self, strs_list, word2idx):
        vectors = []
        for strs in strs_list:
            vec = [word2idx[token] if token in word2idx else 0 for token in strs]
            vectors.append(vec)
        return vectors

    def _vectorize_to_char_tensor(self, config, strs_list, alpha_dict: str):
        """
        Convert query to char-level one hot vector
        Return: [batch, seq_len (word_num), cnn_word_len, 1 + len(alpha_dict)]
        """
        eye_matrix = torch.eye(1 + len(alpha_dict))  # Add 1 is for <pad> char
        cnn_word_len = config['cnn_word_len']
        vectors = []
        for strs in strs_list:
            vector = []
            for str in strs:
                char_vec = [eye_matrix[alpha_dict.find(ch) + 1] for ch in str[:cnn_word_len]] + \
                           [eye_matrix[0] for i in range(max(0, cnn_word_len - len(str)))]
                char_vec = torch.stack(char_vec)
                vector.append(char_vec)
            vectors.append(torch.stack(vector))
        return vectors

    def _compute_lexical_similarity(self, config, query, paragraph, query_to_para_idx):
        from sklearn.feature_extraction.text import TfidfVectorizer
        lexical_sim = []
        vectorizer = TfidfVectorizer()
        for idx, que in enumerate(query):
            para = [' '.join(para) for para in
                    paragraph[query_to_para_idx[idx]: query_to_para_idx[idx + 1]]]  # need a string for vectorizer
            que_para = [' '.join(que)] + para
            que_para_vector = vectorizer.fit_transform(que_para)

            que_vector = torch.FloatTensor(que_para_vector[0].toarray()).to(config['device'])
            para_vector = torch.FloatTensor(que_para_vector[1:].toarray()).to(config['device'])
            sim = torch.mm(que_vector, para_vector.transpose(0, 1)).squeeze()
            sim /= torch.sum(sim)  # normalize
            # sim = F.softmax(sim, dim=-1)
            lexical_sim.append(sim)
        return lexical_sim

    def _rerank_sents_by_sim(self, sents, sim):
        sort_idx = torch.argsort(sim, descending=True)
        rerank_sent = [sents[idx] for idx in sort_idx]
        rerank_sim = [sim[idx].item() for idx in sim]
        return rerank_sent, rerank_sim