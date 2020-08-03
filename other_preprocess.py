# This file is used for other dataset preprocessing, to generate the same format data with training dataset (Trivia QA)

import json
import spacy
import nltk
import trivia_preprocess as tp

def read_bioasq(filepath):
    nlp = spacy.load('en_core_web_sm')
    print('begin to read jsonl files: ', filepath)
    data_dict = []
    ignore_count = 0

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in obj:
                continue

            # Parse the context into several sentences
            sentences_span = []
            sentences_tok = []
            context = obj['context']
            context_sent = nlp(context).sents
            start_span = 0
            for sent in context_sent:
                sentences_tok.append([token.string.lower() for token in nlp(sent.text)])
                sentences_span.append((start_span, start_span + len(sentences_tok[-1])))
                start_span += len(sentences_tok[-1])

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
                sent_visited = [False] * len(sentences_span)
                for answer_info in qa['detected_answers']:
                    for token_span in answer_info['token_spans']:
                        for sent_idx, sent_span in enumerate(sentences_span):
                            if not sent_visited[sent_idx] and \
                            sent_span[0] <= token_span[0] and sent_span[1] >= token_span[1]:
                                sent_visited[sent_idx] = True
                                cur_dict['relevant'].append(sentences_tok[sent_idx])
                                break

                for sent_idx, sent_tok in enumerate(sentences_tok):
                    if not sent_visited[sent_idx]:
                        cur_dict['irrelevant'].append(sent_tok)

            if len(cur_dict['relevant']) > 0 and len(cur_dict['irrelevant']) > 0:
                data_dict.append(cur_dict)
            else:
                ignore_count += 1
    print('discard_count:', ignore_count)
    return data_dict

def read_natural_question(filepath):
    """since the dataset is quite large, we simply use .?! as sentence split token, not using spacy """

    print('begin to read jsonl files: ', filepath)
    data_dict = []
    ignore_count = 0

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in obj:
                continue

            # Parse the context into several sentences
            sentences_span = []
            sentences_tok = []
            context_tok = obj['context_tokens']

            start_span = 0
            sent = []
            for idx, (tok, offset) in enumerate(context_tok):
                sent.append(tok)
                if tok in ['.', '?', '!'] or idx == len(context_tok) - 1:
                    sentences_tok.append(sent)
                    sentences_span.append((start_span, start_span + len(sent)))
                    start_span += len(sent)
                    sent = []

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
                sent_visited = [False] * len(sentences_span)
                for answer_info in qa['detected_answers']:
                    for token_span in answer_info['token_spans']:
                        for sent_idx, sent_span in enumerate(sentences_span):
                            if not sent_visited[sent_idx] and \
                                    sent_span[0] <= token_span[0] and sent_span[1] >= token_span[1]:
                                sent_visited[sent_idx] = True
                                cur_dict['relevant'].append(sentences_tok[sent_idx])
                                break

                for sent_idx, sent_tok in enumerate(sentences_tok):
                    if not sent_visited[sent_idx]:
                        cur_dict['irrelevant'].append(sent_tok)

            if len(cur_dict['relevant']) > 0 and len(cur_dict['irrelevant']) > 0:
                data_dict.append(cur_dict)
            else:
                ignore_count += 1
    print('discard_count:', ignore_count)
    return data_dict

def read_news(filepath):
    return read_natural_question(filepath)

def read_hotpot(filepath):
    return read_natural_question(filepath)

def read_original_hotpot(filepath):
    print('begin to read', filepath)
    with open(filepath, 'r') as f:
        for line in f:
            res = json.loads(line)
    for r in res:
        r['relevant'] = [doc for doc in r['relevant'] if len(doc) > 0]
        r['irrelevant'] = [doc for doc in r['irrelevant'] if len(doc) > 0]
    return res

def convert_original_hotpot_to_format(filepath):
    """ Read original format of HotpotQA (distractor setting) """
    print('begin to read', filepath)
    with open(filepath, 'r') as f:
        for line in f:
            list_of_dict = json.loads(line)

    res = []
    for cur_dict in list_of_dict:
        temp_dict = {
            'qid' : cur_dict['_id'],
            'question_tokens': nltk.word_tokenize(cur_dict['question']),
            'relevant': [],
            'irrelevant': []
        }

        supporting_facts = cur_dict['supporting_facts']
        context = cur_dict['context']
        supporting_facts_set = set([fact[0].lower() + str(fact[1]) for fact in supporting_facts])
        for cont in context:
            title = cont[0].lower()
            for idx, para in enumerate(cont[1]):
                if title + str(idx) in supporting_facts_set:
                    temp_dict['relevant'].append(nltk.word_tokenize(para.lower()))
                else:
                    temp_dict['irrelevant'].append(nltk.word_tokenize(para.lower()))
        res.append(temp_dict)
    return res

if __name__ == '__main__':
    # res = read_natural_question('data/NaturalQuestionsShort.jsonl')
    # res = read_bioasq('data/BioASQ.jsonl')
    # res = tp.read_jsonl_to_list_dict('data/SearchQA.jsonl')
    # res = read_news('data/NewsQA.jsonl')
    res = read_original_hotpot('data/HotpotQA-origin-train.jsonl')
    # tp.data_metric(res)
    for r in res:
        for doc in r['relevant']:
            if len(doc) == 0:   print('x')
        for doc in r['irrelevant']:
            if len(doc) == 0:   print('x')