import time
import torch.nn.functional as F
import torch
import sys

import trivia_preprocess as tp


# ------------------------
# Training phase
# The main difference between Bert and Glove training is how to TOKENIZE and VECTORIZE the text
# ------------------------

def train(config, train_dataloader, dev_dataloader, model, optimizer, word2idx=None, bert_tokenizer=None):
    max_accuracy = 0
    for e in range(1, config['train_epochs'] + 1):
        print('cur_epoch:', e)
        if config['is_train']:
            train_epoch(config, train_dataloader, model, optimizer, word2idx=word2idx, bert_tokenizer=bert_tokenizer)
        similarity, labels = predict(config, dev_dataloader, model, word2idx=word2idx, bert_tokenizer=bert_tokenizer)

        # eval in topk
        accuracy = eval(config, similarity, labels)
        if max_accuracy < accuracy and config['is_train']:
            max_accuracy = accuracy
            torch.save(model.state_dict(), config['model_write_path'])
        if not config['is_train']:
            break
    print('best model top 1:', max_accuracy)


def train_epoch(config, dataloader, model, optimizer, word2idx=None, bert_tokenizer=None):
    model.train()
    model.to(config['device'])
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, data in enumerate(dataloader):
        # List[List[str]], List[List[str]], List[int]
        query, paragraphs, labels, query_to_para_idx = tp.get_query_and_sample_paragraph(data, config)

        # convert to torch tensor
        query_tensor, query_char_tensor, query_len, paragraph_tensor, paragraph_char_tensor, paragraph_len = \
            tp.get_query_para_tensor(config, query, paragraphs, word2idx=word2idx, bert_tokenizer=bert_tokenizer)

        labels = tp.get_label_tensor(config, labels)

        similarity = model.train_forward(query_tensor, query_char_tensor, query_len,
                                         paragraph_tensor, paragraph_char_tensor, paragraph_len,
                                         query_to_para_idx)

        if config['sample_method'] == 'list':
            loss = F.cross_entropy(similarity, labels)
        else:
            # similarity means the prob that first has a higher relevance than the second one (in which case label = 0)
            # so change labels below.
            labels = 1 - labels
            loss = F.binary_cross_entropy(similarity.squeeze(), labels.float())

        running_loss += loss.item()

        if config['use_augment']:
            loss += train_augment_batch(config, data, paragraphs, labels, model, word2idx, bert_tokenizer)
        loss.backward()

        if (batch_idx + 1) % config['accumulate_step'] == 0:
            optimizer.step()
            optimizer.zero_grad()
    end_time = time.time()
    running_loss /= len(dataloader)
    print('Training Loss:', running_loss, 'Time:', end_time - start_time, 's')


def train_augment_batch(config, data, origin_paragraph, origin_label, model, word2idx=None, bert_tokenizer=None):
    """
    The only difference with normal training epoch is to use augmented query to train
    """

    query, paragraphs, labels, query_to_para_idx = tp.get_aug_query_and_sample_paragraph(data, config)
    paragraphs = origin_paragraph
    labels = origin_label

    query_tensor, query_char_tensor, query_len, paragraph_tensor, paragraph_char_tensor, paragraph_len = \
        tp.get_query_para_tensor(config, query, paragraphs, word2idx=word2idx, bert_tokenizer=bert_tokenizer)

    # labels = tp.get_label_tensor(config, labels)

    similarity = model.train_forward(query_tensor, query_char_tensor, query_len,
                                     paragraph_tensor, paragraph_char_tensor, paragraph_len,
                                     query_to_para_idx)

    loss = F.cross_entropy(similarity, labels)
    return loss


# ------------------------
# Predict phase
# ------------------------

def predict(config, dataloader, model, word2idx=None, bert_tokenizer=None):
    print('begin to predict')
    similarity = []
    total_labels = []

    start = time.time()
    exp_num = 0
    with torch.no_grad():
        model.eval()
        model.to(config['device'])
        for batch_idx, data in enumerate(dataloader):
            # List[List[str]], List[List[str]], List[int]
            query, paragraph, labels, query_to_para_idx = tp.get_query_and_paragraph(data, config)
            query_tensor, query_char_tensor, query_len, paragraph_tensor, paragraph_char_tensor, paragraph_len = \
                tp.get_query_para_tensor(config, query, paragraph, word2idx=word2idx, bert_tokenizer=bert_tokenizer)

            sim = model.predict_forward(query_tensor, query_char_tensor, query_len,
                                        paragraph_tensor, paragraph_char_tensor, paragraph_len,
                                        query_to_para_idx)  # List[(para_num)]

            if config['use_lexical']:  # use lexical information by computing tf-idf values
                lexical_sim = tp.compute_lexical_similarity(config, query, paragraph, query_to_para_idx)
                alpha = config['lexical_alpha']
                sim = [alpha * s + (1 - alpha) * ls for s, ls in zip(sim, lexical_sim)]

            similarity.extend(sim)
            total_labels.extend(labels)

            exp_num += len(paragraph_len)

    end = time.time()
    print('Infer time per (query, doc) pair:', (end - start) / exp_num, 's')
    print('Inter time per query:', (end - start) / len(similarity), 's')
    return similarity, total_labels


# ------------------------
# Eval phase
# ------------------------

def eval(config, similarity, labels):
    k1_accuracy = 0
    for k in config['k']:
        accuracy = tp.eval_topk(similarity, labels, k=k)
        precision = tp.eval_precision_topk(similarity, labels, k=k)
        if k == 1:
            k1_accuracy = accuracy
        print('model top', k, ' accuracy evaluation:', accuracy, ', precision evaluation:', precision)
    return k1_accuracy


# ------------------------
# Hyperparameter search
# ------------------------

def alpha_search(config, dataloader, model, word2idx=None, bert_tokenizer=None, pmi_similarity=None):
    total_labels = []
    model_similarity = []
    lexical_similarity = []

    with torch.no_grad():
        model.eval()
        model.to(config['device'])
        for batch_idx, data in enumerate(dataloader):
            # List[List[str]], List[List[str]], List[int]
            query, paragraph, labels, query_to_para_idx = tp.get_query_and_paragraph(data, config)
            query_tensor, query_char_tensor, query_len, paragraph_tensor, paragraph_char_tensor, paragraph_len = \
                tp.get_query_para_tensor(config, query, paragraph, word2idx=word2idx, bert_tokenizer=bert_tokenizer)

            sim = model.predict_forward(query_tensor, query_char_tensor, query_len,
                                        paragraph_tensor, paragraph_char_tensor, paragraph_len,
                                        query_to_para_idx)  # List[(para_num)]
            lexical_sim = tp.compute_lexical_similarity(config, query, paragraph, query_to_para_idx)

            model_similarity.extend(sim)
            total_labels.extend(labels)
            lexical_similarity.extend(lexical_sim)

    print('TF-IDF Lexical Alpha Search...')
    for alpha in range(11):
        alpha /= 10  # range only support int
        final_sim = [alpha * s + (1 - alpha) * ls for s, ls in zip(model_similarity, lexical_similarity)]
        print('cur alpha:', alpha)
        eval(config, final_sim, total_labels)

    if pmi_similarity != None:
        print('PMI Lexical Alpha Search...')
        for alpha in range(11):
            alpha /= 10  # range only support int
            final_sim = [alpha * s + (1 - alpha) * ls for s, ls in zip(model_similarity, pmi_similarity)]
            print('cur alpha:', alpha)
            eval(config, final_sim, total_labels)



