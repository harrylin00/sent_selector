import trivia_preprocess as tp
import other_preprocess as op
from sent_selector import SentSelector
from esim import ESIM
import train
import pmi

import torch.utils.data as data
import torch.optim as optim
import torch
import os
from transformers import BertTokenizer, BertModel, BertConfig, BertTokenizerFast

def set_config(embed_method='glove',
               use_charCNN=False,
               use_lexical=False,
               aggregate_data=False,
               use_esim=False,
               is_train=True,
               use_augment=False,
               sample_method='list'):
    config = {
        'train_data_path': 'data/TriviaQA-train-web.jsonl',
        'dev_data_path': 'data/TriviaQA-dev-web.jsonl',
        'dev_debug_data_path': 'data/TriviaQA-dev-small-web.jsonl',
        'glove_path': 'embeddings/glove.6B.300d.txt',
        'model_write_path': sample_method + '_lstm_'+embed_method+'.pt' if not aggregate_data else 'agg_lstm_'+embed_method+'.pt',
        'model_load_path': sample_method + '_lstm_'+embed_method+'.pt'  if not aggregate_data else 'agg_lstm_'+embed_method+'.pt',

        'embed_method': embed_method,
        'norm_sim': True,
        'bert_config': 'bert-base-uncased',
        'freeze_bert': True,

        'sample_method': sample_method,
        'relevant_num': 1,
        'irrelevant_num': 5,

        'hidden_size': 256,
        'num_layer': 3,
        'dropout': 0.2,
        'weight_decay': 0,    # regularization for weights in the model

        'use_lexical': use_lexical,   # use tf-idf similarity or not
        'lexical_alpha': 0.5,

        'use_charCNN': use_charCNN,
        'cnn_word_len': 20,    # represent the length of each word
        'cnn_input_size': None, # the number of character, decided by alpha_file
        'cnn_hidden_size': 256,
        'alpha_file': 'alphabet.json',
        'alpha_dict': None, # will be gotten from alpha_file

        'aggregate_data': aggregate_data,
        'aggregate_data_filepath': ['data/BioASQ.jsonl',
                                    'data/SearchQA-train.jsonl',
                                    'data/NaturalQuestionsShort-train.jsonl', 
                                    'data/NewsQA-train.jsonl'],

        'use_esim': use_esim,
        'linear_size': 256,

        'use_augment': use_augment,
        'augment_file': 'data/trivia-que.txt',

        'is_load_model': True,
        'is_train': is_train,
        'batch_size': 64,
        'train_epochs': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'accumulate_step': 1,

        'k': [1, 3, 5, 10]
    }

    if use_esim:
        config['model_load_path'] = config['model_load_path'].replace('lstm', 'esim')
        config['model_write_path'] = config['model_write_path'].replace('lstm', 'esim')
    if use_charCNN:
        config = set_charCNN_config(config)
    if use_augment:
        config['model_load_path'] = 'aug_' + config['model_load_path']
        config['model_write_path'] = 'aug_' + config['model_write_path']

    if sample_method == 'pair':
        config['irrelevant_num'] = 1

    return config

def set_charCNN_config(config):
    alphabet = ''.join(tp.read_alpha_dict(config))
    config['alpha_dict'] = alphabet
    config['cnn_input_size'] = len(alphabet) + 1    # add <pad> char
    config['batch_size'] = 4    # charCNN will consume a lot of memory
    config['accumulate_step'] = 16
    config['model_write_path'] = 'cnn_' + config['model_write_path']
    config['model_load_path'] = 'cnn_' + config['model_load_path']
    return config

def load_model_and_optim(config, model):
    if os.path.exists(config['model_load_path']) and config['is_load_model']:
        print('begin to load model:', config['model_load_path'])
        model.load_state_dict(torch.load(config['model_load_path'], map_location=config['device']))

    # optimizer setting
    # When using bert, we will freeze the word embeddings so we should filter it out before feeding into the optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, weight_decay=config['weight_decay'])
    return model, optimizer

def glove_train(config):
    # Glove loading
    word2idx, word_embed = tp.read_glove(config['glove_path'])

    # model setting
    if config['use_esim']:
        model = ESIM(config, word_embed=word_embed)
    else:
        model = SentSelector(config, word_embed=word_embed)

    model, optimizer = load_model_and_optim(config, model)

    return model, optimizer, word2idx, None

def bert_train(config):
    """ use Bert's embeddings """

    # bert info
    tokenizer = BertTokenizer.from_pretrained(config['bert_config'])
    # tokenizer = BertTokenizerFast.from_pretrained(config['bert_config'])
    bert_config = BertConfig().from_pretrained(config['bert_config'])
    word_embed = BertModel.from_pretrained(config['bert_config']).get_input_embeddings()

    if config['use_esim']:
        model = ESIM(config, word_embed=word_embed, bert_config=bert_config)
    else:
        model = SentSelector(config, word_embed=word_embed, bert_config=bert_config)

    model, optimizer = load_model_and_optim(config, model)

    return model, optimizer, None, tokenizer

def lexical_parameter_search():
    config = set_config(embed_method='glove', aggregate_data=True)
    config['is_train'] = False

    dev_dict_list = tp.read_jsonl_to_list_dict(config['dev_data_path'])
    dev_dataloader = data.DataLoader(dev_dict_list, batch_size=config['batch_size'], shuffle=False,
                                     collate_fn=lambda x: x)

    if config['embed_method'] == 'glove':
        model, optimizer, word2idx, tokenizer = glove_train(config)
    elif config['embed_method'] == 'bert':
        model, optimizer, word2idx, tokenizer = bert_train(config)

    # compute pmi score
    pmi_similarity = pmi.compute_pmi(dev_dict_list)
    pmi_similarity = [torch.FloatTensor(sim).to(config['device']) for sim in pmi_similarity]
    pmi_similarity = [sim / torch.sum(sim) for sim in pmi_similarity]

    train.alpha_search(config, dev_dataloader, model, word2idx=word2idx, bert_tokenizer=tokenizer, pmi_similarity=pmi_similarity)

def eval():
    """ eval the model in other dataset except triviaQA"""
    config = set_config(embed_method='bert', use_charCNN=False, use_lexical=True)

    # dict_list = tp.read_jsonl_to_list_dict('data/SearchQA-dev.jsonl')
    # dict_list = op.read_natural_question('data/NaturalQuestionsShort-dev.jsonl')
    dict_list = op.read_hotpot('data/HotpotQA-dev.jsonl')
    dataloader = data.DataLoader(dict_list, batch_size=config['batch_size'], shuffle=False, collate_fn=lambda x:x)

    if config['embed_method'] == 'glove':
        model, optimizer, word2idx, tokenizer = glove_train(config)
    elif config['embed_method'] == 'bert':
        model, optimizer, word2idx, tokenizer = bert_train(config)

    similarity, labels = train.predict(config, dataloader=dataloader, model=model,
                  word2idx=word2idx, bert_tokenizer=tokenizer)
    train.eval(config, similarity, labels)

def debug_main():
    config = set_config(embed_method='glove', use_esim=False, is_train=True, use_lexical=False, sample_method='pair')

    # data loading
    dev_dict_list = tp.read_jsonl_to_list_dict(config['dev_debug_data_path'])
    # dev_dict_list = tp.read_jsonl_to_list_dict(config['dev_data_path'])
    dev_dataloader = data.DataLoader(dev_dict_list, batch_size=config['batch_size'], shuffle=False,
                                     collate_fn=lambda x: x)

    if config['embed_method'] == 'glove':
        model, optimizer, word2idx, tokenizer = glove_train(config)
    elif config['embed_method'] == 'bert':
        model, optimizer, word2idx, tokenizer = bert_train(config)

    # training and predict
    train.train(config, dev_dataloader, dev_dataloader, model, optimizer, word2idx=word2idx, bert_tokenizer=tokenizer)
    # similarity, labels = train.predict(config, dataloader=dev_dataloader, model=model, word2idx=word2idx, bert_tokenizer=tokenizer)
    # similarity = [tp.standardize(sim) for sim in similarity]
    # tp.record_sim('sim.txt', similarity, labels)
    # precision = tp.eval_precision_dynamic(similarity, labels, threshold=0.1)
    # print('precision:', precision)
    # print('accuracy:', tp.eval_accuracy_dynamic(similarity, labels, threshold=0.1))

def train_hotpotqa():
    config = set_config(embed_method='glove')
    config['train_data_path'] = 'data/HotpotQA-origin-train.jsonl'
    config['dev_data_path'] = 'data/HotpotQA-origin-dev.jsonl'
    config['model_load_path'] = 'hotpot_' + config['model_load_path']
    config['model_write_path'] = 'hotpot_' + config['model_write_path']

    train_dict_list = op.read_original_hotpot(config['train_data_path'])
    train_dataloader = data.DataLoader(train_dict_list, batch_size=config['batch_size'], shuffle=True,
                                       collate_fn=lambda x: x)

    dev_dict_list = op.read_original_hotpot(config['dev_data_path'])
    dev_dataloader = data.DataLoader(dev_dict_list, batch_size=config['batch_size'], shuffle=False,
                                     collate_fn=lambda x: x)

    if config['embed_method'] == 'glove':
        model, optimizer, word2idx, tokenizer = glove_train(config)
    elif config['embed_method'] == 'bert':
        model, optimizer, word2idx, tokenizer = bert_train(config)

    # training and predict
    train.train(config, train_dataloader, dev_dataloader, model, optimizer, word2idx=word2idx, bert_tokenizer=tokenizer)

def main():
    config = set_config(embed_method='glove', use_augment=False, sample_method='point')

    # data loading
    train_dict_list = tp.read_jsonl_to_list_dict(config['train_data_path'])
    if config['use_augment']:
        aug_query = tp.read_augment_data(config['augment_file'])
        train_dict_list = tp.add_augment_to_dict_list(train_dict_list, aug_query)
    if config['aggregate_data']:
        train_dict_list.extend(tp.aggregate_data(config))
    train_dataloader = data.DataLoader(train_dict_list, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x:x)

    dev_dict_list = tp.read_jsonl_to_list_dict(config['dev_data_path'])
    dev_dataloader = data.DataLoader(dev_dict_list, batch_size=config['batch_size'], shuffle=False,
                                       collate_fn=lambda x: x)

    if config['embed_method'] == 'glove':
        model, optimizer, word2idx, tokenizer = glove_train(config)
    elif config['embed_method'] == 'bert':
        model, optimizer, word2idx, tokenizer = bert_train(config)

    # training and predict
    train.train(config, train_dataloader, dev_dataloader, model, optimizer, word2idx=word2idx, bert_tokenizer=tokenizer)

if __name__ == '__main__':
    main()
    # lexical_parameter_search()
    # eval()
    # debug_main()
    # train_hotpotqa()