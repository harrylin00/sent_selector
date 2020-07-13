import time
import torch.nn.functional as F
import torch

import trivia_preprocess as tp

#------------------------
# glove training
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
        ks = config['k']
        for k in ks:
            accuracy = tp.eval_topk(similarity, labels, k=k)
            precision = tp.eval_precision_topk(similarity, labels, k=k)
            if k == 1 and max_accuracy < accuracy and config['is_train']:
                max_accuracy = accuracy
                torch.save(model.state_dict(), config['model_write_path'])
            print('model top', k, ' accuracy evaluation:', accuracy, ', precision evaluation:', precision)
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
        query, query_len, paragraphs, paragraphs_len = tp.get_query_para_tensor(config, query, paragraphs,
                                                                                word2idx=word2idx,
                                                                                bert_tokenizer=bert_tokenizer)
        labels = tp.get_label_tensor(config, labels)

        similarity = model.train_forward(query, query_len, paragraphs, paragraphs_len, query_to_para_idx)
        loss = F.cross_entropy(similarity, labels)
        running_loss += loss.item()
        loss.backward()

        if (batch_idx + 1) % config['accumulate_step'] == 0:
            optimizer.step()
            optimizer.zero_grad()
    end_time = time.time()
    running_loss /= len(dataloader)
    print('Training Loss:', running_loss, 'Time:', end_time - start_time, 's')

def predict(config, dataloader, model, word2idx=None, bert_tokenizer=None):
    similarity = []
    total_labels = []

    with torch.no_grad():
        model.eval()
        model.to(config['device'])
        for batch_idx, data in enumerate(dataloader):
            query, paragraph, labels, query_to_para_idx = tp.get_query_and_paragraph(data, config)
            query, query_len, paragraph, paragraph_len = tp.get_query_para_tensor(config, query, paragraph,
                                                                                  word2idx=word2idx,
                                                                                  bert_tokenizer=bert_tokenizer)
            sim = model.predict_forward(query, query_len, paragraph, paragraph_len, query_to_para_idx)   # List[(para_num)]
            similarity.extend(sim)
            total_labels.extend(labels)
    return similarity, total_labels

