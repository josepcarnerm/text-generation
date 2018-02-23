import os, datetime, torch, time, socket, string, math

from config import RESULTS_DIR_LOCAL, RESULTS_DIR_REMOTE

def move(gpu, tensor_list):
    for t in tensor_list:
        t.cuda() if gpu else t.cpu()


def get_savedir(opt):
    # Whenever you add a new model/dataloader, you need to modify this to make sure parameters relying on other
    # models/dataloaders don't appear in the save dir

    ATTR_DONT_INCLUDE_IN_SAVEDIR = ['input_file_train', 'input_file_test', 'seed', 'gpu']

    savedir = RESULTS_DIR_LOCAL if is_local() else RESULTS_DIR_REMOTE
    savedir = savedir + ('/' if not savedir.endswith('/') else '')

    attrs = [attr for attr in dir(opt) if not attr.startswith('_')]
    name = ''
    for attr_name in attrs:
        if attr_name not in ATTR_DONT_INCLUDE_IN_SAVEDIR:
            name += '{}{}={}'.format('-' if name != '' else '',attr_name, getattr(opt, attr_name))

    return savedir + name + '/'


def is_remote():
    return 'nyu' in socket.gethostname()


def is_local():
    return not is_remote()


def log(fname, s, time=''):
    if not os.path.isdir(os.path.dirname(fname)):
            os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    time = '{}:'.format(time) if time != '' else time
    f.write(time + s + '\n')
    f.close()


def zeros(gpu, sizes):
    v = torch.autograd.Variable(torch.zeros(sizes))
    v = v.cuda() if gpu else v.cpu()
    return v


def to_variable(gpu, sentence):
    # Converts a sentence to a pytorch variable of dimension 1

    var = zeros(gpu, [len(sentence)]).long()
    for i,c in enumerate(sentence):
        var[i] = string.printable.index(c)  # Not One-hot encoding: torch.nn.Embeding layer will transform in one-hot internally

    return var


def to_string(variable):
    return ''.join([string.printable[index.data[0]] for index in variable])


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)