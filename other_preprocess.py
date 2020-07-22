# This file is used for other dataset preprocessing, to generate the same format data with training dataset (Trivia QA)

import json
import spacy
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
    # return tp.read_jsonl_to_list_dict(filepath)
    # return read_bioasq(filepath)
    return read_natural_question(filepath)

if __name__ == '__main__':
    # res = read_natural_question('data/NaturalQuestionsShort.jsonl')
    # res = read_bioasq('data/BioASQ.jsonl')
    # res = tp.read_jsonl_to_list_dict('data/SearchQA.jsonl')
    # res = read_news('data/NewsQA.jsonl')
    res = read_hotpot('data/HotpotQA.jsonl')
    tp.data_metric(res)