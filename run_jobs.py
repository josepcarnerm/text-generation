import os
import utils
from config import RESULTS_DIR_LOCAL, RESULTS_DIR_REMOTE

jobs = [
    # {'dataloader': 'single_file_char_tensor', 'model': 'char_rnn_original', 'batch_size': 200, 'lrt': 0.01, 'epoch_size': 100,
    #  'n_epochs': 200, 'hidden_size_rnn': 50, 'n_layers_rnn': 2, 'sentence_len': 200}
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