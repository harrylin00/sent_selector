import trivia_preprocess as tp
from sent_selector import SentSelector
import train

import torch.utils.data as data
import torch.optim as optim
import torch
import os

def set_config():
    return {
        'train_data_path': 'data/TriviaQA-train-web.jsonl',
        'dev_data_path': 'data/TriviaQA-dev-web.jsonl',
        'glove_path': 'embeddings/glove.6B.300d.txt',
        'model_write_path': 'lstm_glove8B.pt',
        'model_load_path': 'lstm_glove8B.pt',
        # 'glove_dict_path': 'embeddings/glove_dict.json',
        # 'glove_embed_path': 'embeddings/glove_embed.npy',

        'embed_method': 'glove',

        'relevant_num': 1,
        'irrelevant_num': 5,

        'input_size': 300,
        'hidden_size': 256,
        'num_layer': 3,
        'dropout': 0.2,

        'is_load_model': True,
        'is_train': True,
        'batch_size': 128,
        'train_epochs': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'accumulate_step': 1,

        'k': [1, 3, 5]
    }

def main():
    config = set_config()

    # data loading
    train_dict_list = tp.read_jsonl_to_list_dict(config['train_data_path'])
    train_dataloader = data.DataLoader(train_dict_list, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x:x)

    dev_dict_list = tp.read_jsonl_to_list_dict(config['dev_data_path'])
    dev_dataloader = data.DataLoader(dev_dict_list, batch_size=config['batch_size'], shuffle=False,
                                       collate_fn=lambda x: x)

    # Glove loading
    word2idx, word_embed = tp.read_glove(config['glove_path'])

    # model setting
    model = SentSelector(config, word_embed)
    if os.path.exists(config['model_load_path']) and config['is_load_model']:
        model.load_state_dict(torch.load(config['model_load_path']))

    # optimizer setting
    optimizer = optim.Adam(model.parameters())

    # training and predict
    train.train(config, train_dataloader, dev_dataloader, model, optimizer, word2idx)

if __name__ == '__main__':
    main()
