# External modules imports
from __future__ import division
import argparse, pdb, os, numpy, time, torch
import random
import traceback

import torch.optim as optim
from torch.utils.data import DataLoader

# Project imports
import utils


# PARAMETERS ----------------------------------------------------------------------------------------------------
#####################
# Training settings #
#####################
parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataloader', type=str, default='multi_file_str')  # Must be a valid file name in dataloaders/ folder
parser.add_argument('-model', type=str, default='word_rnn')  # Must be a valid file name in models/ folder
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=100)
parser.add_argument('-n_epochs', type=int, default=2000)
parser.add_argument('-gpu', type=int, default=1 if utils.is_remote() else 0, help='Which GPU to use, ignored if running in local')
parser.add_argument('-data_dir', type=str, default='data/', help='path for preprocessed dataloader files')
parser.add_argument('-dropout', type=float, default=0.4)

############################
# Model dependent settings #
############################
parser.add_argument('-hidden_size_rnn', type=int, default=200, help='RNN hidden vector size')
parser.add_argument('-n_layers_rnn', type=int, default=2, help='Num layers RNN')
parser.add_argument('-reuse_pred', action='store_true', help='if true, feed prediction in next timestep instead of true input')
parser.add_argument('-use_pretrained_embeddings', action='store_true', help='if true, use pretrained glove embeddings')
parser.add_argument('-bidirectional', action='store_true', help='if true, use bidirectional LSTMs')
parser.add_argument('-glove_dir', type=str, default='data/glove.6B/glove.6B.200d.txt', help='directory to pretrained glove vectors')
parser.add_argument('-char_ngram', type=int, default=2, help='Size of ending char ngram to use in embedding.')
# Word rnn topic dependent parameters
parser.add_argument('-loss_alpha', type=float, default=0.99, help='How much weight reconstruction loss is given over topic closeness loss')
parser.add_argument('-baseline_model', type=str, help='Baseline model from which to pre-init topic model weights')

#################################
# Dataloader dependent settings #
#################################
# Single file
parser.add_argument('-input_file', type=str, default='gutenberg', help='path to input file')
parser.add_argument('-sentence_len', type=int, default=20)

# Multi file
parser.add_argument('-input_folder_path', type=str, default='data_gutenberg', help='path to input file')

opt = parser.parse_args()
opt.data_dir = (opt.data_dir + '/') if not opt.data_dir.endswith('/') else opt.data_dir
# --------------------------------------------------------------------------------------------------------------


# SETTINGS  ----------------------------------------------------------------------------------------------------
torch.manual_seed(opt.seed)
torch.set_default_tensor_type('torch.FloatTensor')

if opt.gpu != 0:
    opt.device = opt.gpu-1
    print('Setting cuda device to {}. ({} Device available)'.format(opt.device, torch.cuda.device_count()))
    print('Options are: {}'.format(opt))
    torch.cuda.set_device(opt.gpu-1)

# Set filename based on parameters
opt.save_dir = utils.get_savedir(opt)
print("Saving to " + opt.save_dir)
# --------------------------------------------------------------------------------------------------------------


# DATALOADER ---------------------------------------------------------------------------------------------------
print('Initializing dataloader...')
mod = __import__('dataloaders.{}'.format(opt.dataloader), fromlist=['MyDataset'])
datasetClass = getattr(mod, 'MyDataset')
train_dataset = datasetClass(opt, train=True)
test_dataset = datasetClass(opt, train=False)
# train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory= utils.is_remote())
# test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory= utils.is_remote())

def get_batch(dataset):
    batch = [['' for _ in range(opt.batch_size)] for _ in range(opt.sentence_len+1)]
    for b in range(opt.batch_size):
        item = dataset.__getitem__(random.randint(0,len(dataset)))
        for i in range(opt.sentence_len+1):
            batch[i][b] = item[i]
    return batch

get_batch(train_dataset)
get_batch(test_dataset)
print('Train dataset loaded with {} sentences of {} words each. {} words in total'.format(len(train_dataset), opt.sentence_len, len(train_dataset)*opt.sentence_len))
print('Test dataset loaded with {} sentences of {} words each. {} words in total'.format(len(test_dataset), opt.sentence_len, len(train_dataset)*opt.sentence_len))
# --------------------------------------------------------------------------------------------------------------


# TRAIN --------------------------------------------------------------------------------------------------------
def train_epoch(epoch):
    total_loss = 0
    model.train()

    for i in range(opt.epoch_size):

        batch = get_batch(train_dataset)
        optimizer.zero_grad()
        model.zero_grad()

        # Forward step
        loss_batch = model.evaluate(batch)
        total_loss += loss_batch.data[0] / opt.sentence_len

        # Backward step
        loss_batch.backward()
        optimizer.step()
        if epoch<10:
            print('[Train] time:{}, iter:{}, loss:{}'.format(utils.time_since(start), i, loss_batch.data[0] / opt.sentence_len))

    if 'analyze' in dir(model):
        model.analyze([sentence[:5] for sentence in get_batch(train_dataset)])

    return total_loss / opt.epoch_size


def test_epoch(epoch):
    total_loss = 0
    model.eval()

    for i in range(opt.epoch_size):
        batch = get_batch(test_dataset)

        # Forward step
        loss_batch = model.evaluate(batch)
        total_loss += loss_batch.data[0] / opt.sentence_len
        if epoch<10:
            print('[Test] time:{}, iter:{}, loss:{}'.format(utils.time_since(start), i, loss_batch.data[0] / opt.sentence_len))
    return total_loss / opt.epoch_size


def train(n_epochs):

    # prepare for saving
    os.system("mkdir -p " + opt.save_dir)

    # training
    best_valid_loss = 1e6
    train_loss, valid_loss = [], []
    for i in range(0, n_epochs):
        train_loss.append(train_epoch(i))
        try:
            valid_loss.append(test_epoch(i))
        except:
            print('Error when testing epoch')
            valid_loss.append(1e6)

        # If model improved, save it
        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            # save
            utils.move(gpu=False, tensor_list=model.submodules)
            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss, 'valid_loss': valid_loss,
                        'optimizer': optimizer, 'opt': opt},
                       opt.save_dir + 'checkpoint')
            utils.move(gpu=utils.is_remote(), tensor_list=model.submodules)

        # Print log string
        log_string = ('iter: {:d}, train_loss: {:0.6f}, valid_loss: {:0.6f}, best_valid_loss: {:0.6f}, lr: {:0.5f}').format(
                      (i+1)*opt.epoch_size, train_loss[-1], valid_loss[-1], best_valid_loss, opt.lrt)
        print(log_string)
        utils.log(opt.save_dir + 'logs.txt', log_string, utils.time_since(start))

        # Print example
        warmup = 'Wh' if opt.model == 'char_rnn' else ['what']
        test_sample = model.test(warmup, opt.sentence_len)
        utils.log(opt.save_dir + 'examples.txt', test_sample)
        try:
            print(test_sample + '\n')
        except:
            traceback.print_exc()

# --------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    start = time.time()

    if os.path.isfile(opt.save_dir + 'checkpoint'):
        print('loading existing model...')
        utils.log(opt.save_dir + 'logs.txt', '[loading existing model]')
        checkpoint = torch.load(opt.save_dir + 'checkpoint')
        model = checkpoint.get('model')
        optimizer = checkpoint.get('optimizer')
    else:
        print('Initializing model...')
        mod = __import__('models.{}'.format(opt.model), fromlist=['Model'])
        model = getattr(mod, 'Model')(opt)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, opt.lrt)
        if opt.baseline_model and 'initialize' in dir(model):
            # If baseline model to preinit is provided, use it
            print('Initializing with baseline model')
            model.initialize(opt.baseline_model)

    model = model.cuda() if utils.is_remote() else model

    print('training...')
    utils.log(opt.save_dir + 'logs.txt', '[training]')
    train(opt.n_epochs)

