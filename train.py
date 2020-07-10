import time
import torch.nn.functional as F

import trivia_preprocess as tp

def train(config, train_dataloader, dev_dataloader, model, optimizer, word2idx):
    for e in range(1, config['train_epochs'] + 1):
        print('cur_epoch:', e)
        train_epoch(config, train_dataloader, model, optimizer, word2idx)

def train_epoch(config, dataloader, model, optimizer, word2idx):
    model.train()
    model.to(config['device'])
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, data in enumerate(dataloader):
        # query = tp.get_query(data)  # List[List[str]]
        query, paragraphs, labels, query_to_para_idx = tp.get_query_and_sample_paragraph(data, config)  # List[List[str]], List[List[str]], List[int]

        # convert to torch tensor
        query, query_len, paragraphs, paragraphs_len = tp.get_query_para_tensor(config, query, paragraphs, word2idx)
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