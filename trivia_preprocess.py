import json
import random
import os
from torch.nn.utils.rnn import *
import numpy as np
import torch.nn.functional as F

# ------------------
# read and write data
# ------------------

def read_jsonl_to_list_dict(filepath):
    """
    read jsonl files and convert it to a list of dict
    return dict pattern: {questionId: {'question_tokens': List[str]
                               'relevant': List[List[str]]
                               'irrelevant': List[List[str]]}}

    """
    print('begin to read jsonl files: ', filepath)

    data_dict = []

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in obj:
                continue

            # Parse the context into several paragraphs
            # paragraphs = List[List[str]], paragraph_offset = List[[start_offset, end_offset)]
            paragraphs = []
            paragraphs_offset = []
            para = []
            prev_offset = 0
            for token, offset in obj['context_tokens']:
                if token == '[TLE]' or token == '[DOC]' or token == '[PAR]':
                    if len(para) != 0:
                        paragraphs.append(para)
                        paragraphs_offset.append((prev_offset, offset))
                    para = []
                    prev_offset = offset
                else:
                    para.append(token)
            if len(para) != 0:  # process the last sentence
                paragraphs.append(para)
                paragraphs_offset.append((prev_offset, len(token) + offset))

            # Iterate questions:
            for qa in obj['qas']:
                qid = qa['qid']
                question_tokens = [token.lower() for token, offset, in qa['question_tokens']]

                # build dict
                cur_dict = {'qid': qid,
                            'question_tokens': question_tokens,
                            'relevant': [],
                            'irrelevant': []}

                # get relevant paragraphs by iterating the answers
                paragraph_visited = [False] * len(paragraphs)
                for answer_info in qa['detected_answers']:
                    for char_start, char_end in answer_info['char_spans']:
                        for para_idx, (para_start, para_end) in enumerate(paragraphs_offset):
                            if not paragraph_visited[para_idx] and \
                                para_start <= char_start and para_end >= char_end and \
                                    len(paragraphs[para_idx]) > 0:

                                cur_dict['relevant'].append([word.lower() for word in paragraphs[para_idx]])
                                paragraph_visited[para_idx] = True
                                break
                            if para_start > char_end:
                                break
                for para_idx, para in enumerate(paragraphs):
                    if not paragraph_visited[para_idx] and len(para) > 0:
                        cur_dict['irrelevant'].append([word.lower() for word in para])

                if len(cur_dict['relevant']) > 0 and len(cur_dict['irrelevant']) > 0:
                    data_dict.append(cur_dict)

    return data_dict

# ------------------
# training phase
# ------------------

def get_query_and_sample_paragraph(dict_list, config):
    query = get_query(dict_list)
    query, paragraphs, labels, query_to_para_idx = sample_paragraph(query, dict_list, config)
    return query, paragraphs, labels, query_to_para_idx

def sample_paragraph(query, dict_list, config):
    """
    sample paragraphs from the data, return the list of the paragraph and the relevant para's idx
    relevant_num should not be larger than 1
    """

    relevant_num = config['relevant_num']
    irrelevant_num = config['irrelevant_num']

    labels = []
    paragraphs = []
    real_query = []
    for i, dict in enumerate(dict_list):
        rel_para = dict['relevant']
        irrel_para = dict['irrelevant']

        # ignore the data contains too less relevant/irrelevant paragraphs
        if len(rel_para) < relevant_num or len(irrel_para) < irrelevant_num:
            continue

        # sample the paragraphs
        rel_ids = random.sample(range(len(rel_para)), relevant_num)
        sampled_rel_para = [rel_para[idx] for idx in rel_ids]
        irrel_ids = random.sample(range(len(irrel_para)), irrelevant_num)
        sampled_irrel_para = [irrel_para[idx] for idx in irrel_ids]
        sampled_para = sampled_rel_para + sampled_irrel_para

        # shuffle the paragraphs, not sure if it is necessary
        label_list = [0] * len(sampled_para)
        label_list[0] = 1
        combined = list(zip(label_list, sampled_para))
        random.shuffle(combined)
        label_list, sampled_para = zip(*combined)

        labels.append(label_list.index(1))  # find the relevant para's idx
        paragraphs.extend(sampled_para)
        real_query.append(query[i])

    return real_query, paragraphs, labels, list(range(0, len(paragraphs) + 1, relevant_num + irrelevant_num))

def get_query(dict_list):
    return [dict['question_tokens'] for dict in dict_list]

# ------------------
# predict phase
# ------------------

def get_query_and_paragraph(dict_list, config):
    query = get_query(dict_list)
    paragraph, labels, query_to_para_idx = get_paragraph(dict_list, config)
    return query, paragraph, labels, query_to_para_idx

def get_paragraph(dict_list, config):
    labels = []
    paragraphs = []
    query_to_para_idx = []
    for dict in dict_list:
        query_to_para_idx.append(len(paragraphs))
        paragraphs.extend(dict['relevant'])
        paragraphs.extend(dict['irrelevant'])
        cur_label = [1] * len(dict['relevant']) + [0] * len(dict['irrelevant'])
        labels.append(cur_label)
    query_to_para_idx.append(len(paragraphs))
    return paragraphs, labels, query_to_para_idx

# ------------------
# eval phase
# ------------------

def eval_topk(similarity, labels, k):
    accuracy = []
    for sim, label in zip(similarity, labels):
        if len(label) <= k or sum(label) == 0:
            continue
        sort_idx = torch.argsort(sim, descending=True)
        topk_list = [label[idx] for i, idx in enumerate(sort_idx[:k])]
        accuracy.append(max(topk_list))
    return np.mean(accuracy)

def eval_precision_topk(similarity, labels, k):
    precision = []
    for sim, label in zip(similarity, labels):
        if len(label) <= k or sum(label) == 0:
            continue
        sort_idx = torch.argsort(sim, descending=True)
        topk_list = [label[idx] for i, idx in enumerate(sort_idx[:min(k, sum(label))])]
        precision.append(np.mean(topk_list))
    return np.mean(precision)

# ------------------
# dictionary building
# ------------------

def read_glove(filepath):
    """
    get word2idx and word_embed
    NOTE: set <PAD> as word_idx = 0, embed_size is 300d
    """
    print('reading glove files:', filepath)

    word2idx = {}
    word_embed = [[0] * 300]    # word_embed[0] = [0] * 300, represent the <PAD>

    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            line_list = line.split()
            word = ' '.join(line_list[: len(line_list)-300])
            embed = [float(num) for num in line_list[len(line_list)-300:]]

            word2idx[word] = idx + 1
            word_embed.append(embed)

    return word2idx, word_embed

# ------------------
# utility function
# ------------------

def get_label_tensor(config, labels):
    return torch.LongTensor(labels).to(config['device'])

def get_query_para_tensor(config, query, paragraphs, word2idx=None, bert_tokenizer=None):
    """ encode query and paragraphs into vector with tensor version"""
    if config['embed_method'] == 'bert':
        # tokenize the query and paragraphs
        query = bert_tokenize(query, bert_tokenizer)
        paragraphs = bert_tokenize(paragraphs, bert_tokenizer)

        # vectorize, simply choose [1:-1] because of removing [CLS] and [SEP]
        # bert is restricted with max_len=512
        query = [bert_tokenizer.encode(que, max_length=512, truncation=True)[1:-1] for que in query]
        paragraphs = [bert_tokenizer.encode(para, max_length=512, truncation=True)[1:-1] for para in paragraphs]
    elif config['embed_method'] == 'glove':
        query = vectorize_to_tensor(query, word2idx=word2idx)
        paragraphs = vectorize_to_tensor(paragraphs, word2idx=word2idx)

    query = [torch.LongTensor(q) for q in query]
    query_len = torch.LongTensor([len(q) for q in query])
    query_pad = pad_sequence(query, batch_first=True)

    paragraphs = [torch.LongTensor(para) for para in paragraphs]
    paragraphs_len = torch.LongTensor([len(para) for para in paragraphs])
    paragraphs_pad = pad_sequence(paragraphs, batch_first=True)

    device = config['device']
    return query_pad.to(device), query_len.to(device), paragraphs_pad.to(device), paragraphs_len.to(device)

def vectorize_to_tensor(strs_list, word2idx):
    vectors = []
    for strs in strs_list:
        vec = [word2idx[token] if token in word2idx else 0 for token in strs]
        vectors.append(vec)
    return vectors

def data_metric(data_dict):
    """return some description metric of the data"""

    question_num = 0
    relevant_num = 0
    irrelevant_num = 0
    para_num = 0
    para_len = 0
    for dict in data_dict:
        question_num += 1
        relevant_num += len(dict['relevant'])
        irrelevant_num += len(dict['irrelevant'])
        para_num += len(dict['relevant']) + len(dict['irrelevant'])
        para_len += sum([len(para) for para in dict['relevant']])
        para_len += sum([len(para) for para in dict['irrelevant']])

    print('question_num: ', question_num)
    print('para_num per question', para_num / question_num)
    print('relevant_num per question: ', relevant_num / question_num)
    print('irrelevant_num per question: ', irrelevant_num / question_num)
    print('para_length per para: ', para_len / para_num)

# ------------------
# Bert utility function
# ------------------

def bert_tokenize(list_strs, bert_tokenizer):
    res = []
    for strs in list_strs:
        tokenized_res = []
        for token in strs:
            tokenized_res.extend(bert_tokenizer.tokenize(token))
        if len(tokenized_res) == 0:     # remove the case when tokenized_res = [], else it will raise an error when encoding
            tokenized_res = ['']
        res.append(tokenized_res)
    return res

# --------------------
# Lexical Vectors / Similarity
# --------------------

def compute_lexical_similarity(config, query, paragraph, query_to_para_idx):
    from sklearn.feature_extraction.text import TfidfVectorizer
    lexical_sim = []
    vectorizer = TfidfVectorizer()
    for idx, que in enumerate(query):
        para = [' '.join(para) for para in paragraph[query_to_para_idx[idx]: query_to_para_idx[idx + 1]]] # need a string for vectorizer
        que_para = [' '.join(que)] + para
        que_para_vector = vectorizer.fit_transform(que_para)

        que_vector = torch.FloatTensor(que_para_vector[0].toarray()).to(config['device'])
        para_vector = torch.FloatTensor(que_para_vector[1:].toarray()).to(config['device'])
        sim = torch.mm(que_vector, para_vector.transpose(0, 1)).squeeze()
        sim /= torch.sum(sim)   # normalize
        # sim = F.softmax(sim, dim=-1)
        lexical_sim.append(sim)
    return lexical_sim

