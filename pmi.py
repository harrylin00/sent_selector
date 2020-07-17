import trivia_preprocess as tp
import other_preprocess as op

from collections import defaultdict
import math
import torch

def build_index(docs):
    index = defaultdict(lambda : defaultdict(list))
    for doc_idx, doc in enumerate(docs):
        for gram_idx, gram in enumerate(doc):
            index[gram][doc_idx].append(gram_idx)
    return index

def num_occurrences(gram: str, index) -> int:
    """ Get the number of sentences in which a gram appears """
    return len(index.get(gram, {}))

def num_co_occurrences(gram1: str, gram2: str, index: defaultdict, within: int) -> int:
    """ Get the number of sentences in which two grams occur closely """

    index1 = index.get(gram1, {})
    index2 = index.get(gram2, {})

    return len([
        sentence
        for sentence, locs in index1.items()
        if any(abs(loc1 - loc2) <= within
            for loc1 in locs
            for loc2 in index2.get(sentence, []))
    ])

def pmi(prob_x: float, prob_y: float, prob_xy: float) -> float:
    """ Calculate pmi using probabilities, return 0.0 if p_xy is 0 """

    if prob_xy > 0.0:
        return math.log(prob_xy / prob_x / prob_y)
    else:
        return 0.0

def count_pmi(num_x: int, num_y: int, num_xy: int, doc_num) -> float:
    """ Calculate pmi using counts """
    return pmi(num_x / doc_num, num_y / doc_num, num_xy / doc_num)

def compute_pmi(dict_list):
    """
    Compute the pmi of the dict_list
    Return List[List[float (score) ]]
    """
    pmi_score = []
    for dict in dict_list:
        docs = dict['relevant'] + dict['irrelevant']
        cur_pmi_score = torch.FloatTensor([0] * len(docs))
        doc_num = len(docs)
        index = build_index(docs)
        question_tok = dict['question_tokens']
        q_counts = [num_occurrences(token, index) for token in question_tok]
        for doc_idx, doc in enumerate(docs):
            total_pmi = 0.0
            count = 0
            doc = list(set(doc))
            for doc_gram in doc:
                doc_gram_count = num_occurrences(doc_gram, index)
                for ques_gram, ques_gram_count in zip(question_tok, q_counts):
                    co_count = num_co_occurrences(doc_gram, ques_gram, index, within=10)
                    cpmi = count_pmi(doc_gram_count, ques_gram_count, co_count, doc_num)
                    total_pmi += cpmi
                    count += 1
            if count > 0:
                cur_pmi_score[doc_idx] = total_pmi / count
        pmi_score.append(cur_pmi_score)
    return pmi_score

def get_labels(dict_list):
    return [[1] * len(dict['relevant']) + [0] * len(dict['irrelevant']) for dict in dict_list]

def main():
    dev_data_path = 'data/TriviaQA-dev-web.jsonl'
    # dev_data_path = 'data/SearchQA-dev.jsonl'
    dict_list = tp.read_jsonl_to_list_dict(dev_data_path)
    # dev_data_path = 'data/NaturalQuestionsShort-dev.jsonl'
    # dict_list = op.read_natural_question(dev_data_path)

    pmi_score = compute_pmi(dict_list)
    labels = get_labels(dict_list)
    ks = [1, 3, 5, 10]
    for k in ks:
        accuracy = tp.eval_topk(pmi_score, labels, k=k)
        precision = tp.eval_precision_topk(pmi_score, labels, k=k)
        print('pmi top', k,'accuracy: ', accuracy, 'precision: ', precision)

if __name__ == '__main__':
    main()