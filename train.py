import time
import torch.nn.functional as F
import torch
import sys

import trivia_preprocess as tp

#------------------------
# Training phase
# The main difference between Bert and Glove training is how to TOKENIZE and VECTORIZE the text
#------------------------

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
        
        try:
            loss = F.cross_entropy(similarity, labels)
        except:
            print('batch_idx:', batch_idx)
            print('similarity shape:', similarity.shape)
            print('label shape:', labels.shape)
            sys.exit(0)

        running_loss += loss.item()
        loss.backward()

        if (batch_idx + 1) % config['accumulate_step'] == 0:
            optimizer.step()
            optimizer.zero_grad()
    end_time = time.time()
    running_loss /= len(dataloader)
    print('Training Loss:', running_loss, 'Time:', end_time - start_time, 's')

#------------------------
# Predict phase
#------------------------

def predict(config, dataloader, model, word2idx=None, bert_tokenizer=None):
    similarity = []
    total_labels = []

    with torch.no_grad():
        model.eval()
        model.to(config['device'])
        for batch_idx, data in enumerate(dataloader):
            # List[List[str]], List[List[str]], List[int]
            query, paragraph, labels, query_to_para_idx = tp.get_query_and_paragraph(data, config)
            query_tensor, query_char_tensor, query_len, paragraph_tensor, paragraph_char_tensor, paragraph_len =\
                tp.get_query_para_tensor(config, query, paragraph, word2idx=word2idx, bert_tokenizer=bert_tokenizer)

            sim = model.predict_forward(query_tensor, query_char_tensor, query_len,
                                        paragraph_tensor, paragraph_char_tensor, paragraph_len,
                                        query_to_para_idx)   # List[(para_num)]

            if config['use_lexical']:    # use lexical information by computing tf-idf values
                lexical_sim = tp.compute_lexical_similarity(config, query, paragraph, query_to_para_idx)
                alpha = config['lexical_alpha']
                sim = [alpha * s + (1 - alpha) * ls for s, ls in zip(sim, lexical_sim)]

            similarity.extend(sim)
            total_labels.extend(labels)
    return similarity, total_labels

#------------------------
# Eval phase
#------------------------

def eval(config, similarity, labels):
    k1_accuracy = 0
    for k in config['k']:
        accuracy = tp.eval_topk(similarity, labels, k=k)
        precision = tp.eval_precision_topk(similarity, labels, k=k)
        if k == 1:
            k1_accuracy = accuracy
        print('model top', k, ' accuracy evaluation:', accuracy, ', precision evaluation:', precision)
    return k1_accuracy

#------------------------
# Hyperparameter search
#------------------------

def alpha_search(config, dataloader, model, word2idx=None, bert_tokenizer=None):
    total_labels = []
    model_similarity = []
    lexical_similarity = []

    with torch.no_grad():
        model.eval()
        model.to(config['device'])
        for batch_idx, data in enumerate(dataloader):
            # List[List[str]], List[List[str]], List[int]
            query, paragraph, labels, query_to_para_idx = tp.get_query_and_paragraph(data, config)
            query_tensor, query_char_tensor, query_len, paragraph_tensor, paragraph_char_tensor, paragraph_len =\
                tp.get_query_para_tensor(config, query, paragraph, word2idx=word2idx, bert_tokenizer=bert_tokenizer)

            sim = model.predict_forward(query_tensor, query_char_tensor, query_len,
                                        paragraph_tensor, paragraph_char_tensor, paragraph_len,
                                        query_to_para_idx)   # List[(para_num)]
            lexical_sim = tp.compute_lexical_similarity(config, query, paragraph, query_to_para_idx)

            model_similarity.extend(sim)
            total_labels.extend(labels)
            lexical_similarity.extend(lexical_sim)

    for alpha in range(11):
        alpha /= 10         # range only support int
        final_sim = [alpha * s + (1 - alpha) * ls for s, ls in zip(model_similarity, lexical_similarity)]
        print('cur alpha:', alpha)
        eval(config, final_sim, total_labels)



