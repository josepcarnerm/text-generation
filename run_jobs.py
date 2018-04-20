import os
import utils
from config import RESULTS_DIR_LOCAL, RESULTS_DIR_REMOTE

jobs = [

    {'dataloader': 'multi_file_str', 'model': 'word_rnn', 'batch_size': 128, 'lrt': 0.0001,
     'epoch_size': 1000, 'n_epochs': 500, 'hidden_size_rnn': 200, 'n_layers_rnn': 2, 'sentence_len': 20,
     'reuse_pred': '', 'use_pretrained_embeddings': ''},
    {'dataloader': 'multi_file_str', 'model': 'word_rnn', 'batch_size': 128, 'lrt': 0.0001,
     'epoch_size': 1000, 'n_epochs': 500, 'hidden_size_rnn': 200, 'n_layers_rnn': 4, 'sentence_len': 20,
     'reuse_pred': '', 'use_pretrained_embeddings': ''},
    {'dataloader': 'multi_file_str', 'model': 'word_rnn', 'batch_size': 128, 'lrt': 0.0001,
     'epoch_size': 1000, 'n_epochs': 500, 'hidden_size_rnn': 100, 'n_layers_rnn': 2, 'sentence_len': 20,
     'reuse_pred': '', 'use_pretrained_embeddings': ''},
    {'dataloader': 'multi_file_str', 'model': 'word_rnn', 'batch_size': 128, 'lrt': 0.0001,
     'epoch_size': 1000, 'n_epochs': 500, 'hidden_size_rnn': 300, 'n_layers_rnn': 2, 'sentence_len': 20,
     'reuse_pred': '', 'use_pretrained_embeddings': ''},
    {'dataloader': 'multi_file_str', 'model': 'word_rnn', 'batch_size': 128, 'lrt': 0.0001,
     'epoch_size': 1000, 'n_epochs': 500, 'hidden_size_rnn': 200, 'n_layers_rnn': 2, 'sentence_len': 20,
     'reuse_pred': ''},
    {'dataloader': 'multi_file_str', 'model': 'word_rnn', 'batch_size': 128, 'lrt': 0.0001,
     'epoch_size': 1000, 'n_epochs': 500, 'hidden_size_rnn': 200, 'n_layers_rnn': 2, 'sentence_len': 20,
     'use_pretrained_embeddings': ''},

]

srun_args = ['']
save_dir = RESULTS_DIR_LOCAL if utils.is_local() else RESULTS_DIR_REMOTE


def run_job(job):

    name = ''
    for k, v in sorted(job.items()):
        name += ('' if name == '' else '-') + ('{}={}'.format(k,v) if v!="" else k)

    command_srun = (
        'srun --job-name "{}" --output "{}/{}.out" --err "{}/{}.err" --mail-type=ALL --mail-user=jcm807@nyu.edu '
        '--gres=gpu:1 --qos=batch --nodes=1 --constraint=gpu_12gb'
    ).format(name, save_dir, name, save_dir, name)

    command_py = 'python train.py '
    for k,v in sorted(job.items()):
        command_py += '-{} "{}" '.format(k,v) if v!="" else '-{} '.format(k)

    command = command_py if utils.is_local() else command_srun + ' ' + command_py + '&'
    print('Running command: '+command)
    os.system(command)

for job in jobs:
    run_job(job)