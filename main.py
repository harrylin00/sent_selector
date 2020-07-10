import trivia_preprocess as tp
from sent_selector import SentSelector
import train

import torch.utils.data as data
import torch.optim as optim
import torch
import os
from transformers import BertTokenizer, BertModel, BertConfig

def set_config(embed_method='glove'):
    return {
        'train_data_path': 'data/TriviaQA-train-web.jsonl',
        'dev_data_path': 'data/TriviaQA-dev-web.jsonl',
        'glove_path': 'embeddings/glove.6B.300d.txt',
        'model_write_path': 'lstm_'+embed_method+'.pt',
        'model_load_path': 'lstm_'+embed_method+'.pt',
        # 'glove_dict_path': 'embeddings/glove_dict.json',
        # 'glove_embed_path': 'embeddings/glove_embed.npy',

        'embed_method': embed_method,
        'bert_config': 'bert-base-uncased',

        'relevant_num': 1,
        'irrelevant_num': 5,

        # 'input_size': 300,
        'hidden_size': 512,
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

def glove_train(config, train_dataloader, dev_dataloader):
    # Glove loading
    word2idx, word_embed = tp.read_glove(config['glove_path'])

    # model setting
    model = SentSelector(config, word_embed=word_embed)
    if os.path.exists(config['model_load_path']) and config['is_load_model']:
        model.load_state_dict(torch.load(config['model_load_path']))

    # optimizer setting
    # When using bert, we will freeze the word embeddings so we should filter it out before feeding into the optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters)

    # training and predict
    train.train(config, train_dataloader, dev_dataloader, model, optimizer, word2idx=word2idx)

def bert_train(config, train_dataloader, dev_dataloader):
    """ use Bert's embeddings """

    # bert info
    tokenizer = BertTokenizer.from_pretrained(config['bert_config'])
    bert_config = BertConfig().from_pretrained(config['bert_config'])
    word_embed = BertModel.from_pretrained(config['bert_config']).get_input_embeddings()

    model = SentSelector(config, word_embed=word_embed, bert_config=bert_config)
    if os.path.exists(config['model_load_path']) and config['is_load_model']:
        model.load_state_dict(torch.load(config['model_load_path']))

    # optimizer setting
    # When using bert, we will freeze the word embeddings so we should filter it out before feeding into the optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters)

    train.train(config, train_dataloader, dev_dataloader, model, optimizer, bert_tokenizer=tokenizer)


def main():
    config = set_config()

    # data loading
    train_dict_list = tp.read_jsonl_to_list_dict(config['train_data_path'])
    train_dataloader = data.DataLoader(train_dict_list, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x:x)

    dev_dict_list = tp.read_jsonl_to_list_dict(config['dev_data_path'])
    dev_dataloader = data.DataLoader(dev_dict_list, batch_size=config['batch_size'], shuffle=False,
                                       collate_fn=lambda x: x)

    if config['embed_method'] == 'glove':
        glove_train(config, train_dataloader, dev_dataloader)
    elif config['embed_method'] == 'bert':
        bert_train(config, train_dataloader, dev_dataloader)

if __name__ == '__main__':
    main()
