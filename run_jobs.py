import os
import utils
from config import RESULTS_DIR_LOCAL, RESULTS_DIR_REMOTE

jobs = [
    {'dataloader': 'single_file', 'model': 'char_rnn', 'batch_size': 64, 'lrt': 0.0005, 'epoch_size': 20,
     'n_epochs': 2000, 'hidden_size_rnn': 100, 'n_layers_rnn': 2, 'sentence_len': 20}
]

srun_args = ['']
save_dir = RESULTS_DIR_LOCAL if utils.is_local() else RESULTS_DIR_REMOTE


def run_job(job):

    name = ''
    for k, v in sorted(job.items()):
        name += ('' if name == '' else '-') + '{}={}'.format(k,v)

    command_srun = (
        'srun --job-name "{}" --output "{}/{}.out" --err "{}/{}.err" --mail-type=ALL --mail-user=jcm807@nyu.edu '
        '--gres=gpu:1 --qos=batch --nodes=1 --constraint=gpu_12gb'
    ).format(name, save_dir, name, save_dir, name)

    command_py = 'python train.py '
    for k,v in sorted(job.items()):
        command_py += '-{} "{}" '.format(k,v)

    command = command_py if utils.is_local() else command_srun + ' ' + command_py + '&'
    print('Running command: '+command)
    os.system(command)

for job in jobs:
    run_job(job)