# External modules imports
from __future__ import division
import argparse, pdb, os, numpy, time, torch
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
parser.add_argument('-dataloader', type=str, default='sep_top_file_words')  # Must be a valid file name in dataloaders/ folder
parser.add_argument('-model', type=str, default='word_rnn_sep_top_file')  # Must be a valid file name in models/ folder
parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-lrt', type=float, default=0.01)
parser.add_argument('-epoch_size', type=int, default=100)
parser.add_argument('-n_epochs', type=int, default=200)
parser.add_argument('-gpu', type=int, default=1 if utils.is_remote() else 0, help='Which GPU to use, ignored if running in local')

############################
# Model dependent settings #
############################
parser.add_argument('-hidden_size_rnn', type=int, default=100, help='RNN hidden vector size')
parser.add_argument('-n_layers_rnn', type=int, default=2, help='Num layers RNN')
# Word rnn topic dependent parameters
parser.add_argument('-loss_alpha', type=float, default=0.5, help='How much weight reconstruction loss is given over topic closeness loss')

#################################
# Dataloader dependent settings #
#################################
parser.add_argument('-input_file_train', type=str, default='data/shakespeare_train.txt', help='path to input file for training data')
parser.add_argument('-input_file_test', type=str, default='data/shakespeare_test.txt', help='path to input file for test data')
parser.add_argument('-sentence_len', type=int, default=20)
parser.add_argument('-glove_dir', type=str, default='data/glove.6B/glove.6B.100d.txt', help='directory to pretrained glove vectors')

opt = parser.parse_args()
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

train_dataloader = DataLoader(datasetClass(opt, train=True), batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory= utils.is_remote())
test_dataloader = DataLoader(datasetClass(opt, train=False), batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory= utils.is_remote())
# --------------------------------------------------------------------------------------------------------------


# TRAIN --------------------------------------------------------------------------------------------------------
def train_epoch(nsteps):
    total_loss = 0
    model.train()
    for iter, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        model.zero_grad()

        # Forward step
        loss_batch = model.evaluate(batch)
        total_loss += loss_batch.data[0] / opt.sentence_len

        # Backward step
        loss_batch.backward()
        optimizer.step()

        if iter == nsteps:
            break
    return total_loss / nsteps


def test_epoch(nsteps):
    total_loss = 0
    model.eval()
    for iter, batch in enumerate(test_dataloader):

        # Forward step
        loss_batch = model.evaluate(batch)
        total_loss += loss_batch.data[0] / opt.sentence_len

        if iter == nsteps:
            break

    return total_loss / nsteps


def train(n_epochs):

    # prepare for saving
    os.system("mkdir -p " + opt.save_dir)

    # training
    best_valid_loss = 1e6
    train_loss, valid_loss = [], []
    for i in range(0, n_epochs):
        train_loss.append(train_epoch(opt.epoch_size))
        valid_loss.append(test_epoch(opt.epoch_size))

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
        if opt.model == 'word_rnn_topic':
            str_debug = 'Average reconstruction loss: {}, average topic closeness loss: {}'.format(
                model.get_avg_losses()[0], model.get_avg_losses()[1]
            )
            utils.log(opt.save_dir + 'logs.txt', str_debug, utils.time_since(start))
        print(log_string)
        utils.log(opt.save_dir + 'logs.txt', log_string, utils.time_since(start))

        # Print example
        warmup = 'Wh' if opt.model == 'char_rnn' else ['What']
        test_sample = model.test(warmup, opt.sentence_len)
        utils.log(opt.save_dir + 'examples.txt', test_sample)
        print(test_sample + '\n')

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

    model = model.cuda() if utils.is_remote() else model

    print('training...')
    utils.log(opt.save_dir + 'logs.txt', '[training]')
    train(opt.n_epochs)

