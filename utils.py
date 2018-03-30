import os, csv, torch, time, socket, string, math
import numpy as np
from textblob import TextBlob as tb

from config import RESULTS_DIR_LOCAL, RESULTS_DIR_REMOTE

N_CHARS = len(string.printable)
ALL_CHARS = string.printable


def move(gpu, tensor_list):
    for t in tensor_list:
        t.cuda() if gpu else t.cpu()

def get_savedir(opt):
    # Whenever you add a new model/dataloader, you need to modify this to make sure parameters relying on other
    # models/dataloaders don't appear in the save dir

    ATTR_DONT_INCLUDE_IN_SAVEDIR = ['input_file_train', 'input_file_test', 'seed', 'gpu']

    if opt.model != 'word_rnn_topic':
        ATTR_DONT_INCLUDE_IN_SAVEDIR.append('loss_alpha')

    savedir = RESULTS_DIR_LOCAL if is_local() else RESULTS_DIR_REMOTE
    savedir = savedir + ('/' if not savedir.endswith('/') else '')

    attrs = [attr for attr in dir(opt) if not attr.startswith('_')]
    name = ''
    for attr_name in attrs:
        if attr_name not in ATTR_DONT_INCLUDE_IN_SAVEDIR:
            name += '{}{}={}'.format('-' if name != '' else '',attr_name, getattr(opt, attr_name))

    return savedir + name + '/'


def is_remote():
    return 'nyu' in socket.gethostname() or 'matt-Z170XP-SLI' == socket.gethostname()


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


# def to_variable(gpu, sentence):
# #     # Converts a sentence to a pytorch variable of dimension 1
# #
# #     var = zeros(gpu, [len(sentence)]).long()
# #     for i,c in enumerate(sentence):
# #         try:
# #             var[i] = string.printable.index(c)  # Not One-hot encoding: torch.nn.Embeding layer will transform in one-hot internally
# #         except:
# #             var[i] = string.printable.index(' ')
# #
# #     return var
# #
# #
# # def to_string(variable):
# #     return ''.join([string.printable[index.data[0]] for index in variable])


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):
    """Reads in matrices from CSV or space-delimited files.
    Parameters
    ----------
    src_filename : str
        Full path to the file to read.
    delimiter : str (default: ',')
        Delimiter for fields in src_filename. Use delimter=' '
        for GloVe files.
    header : bool (default: True)
        Whether the file's first row contains column names.
        Use header=False for GloVe files.
    quoting : csv style (default: QUOTE_MINIMAL)
        Use the default for normal csv files and csv.QUOTE_NONE for
        GloVe files.
    Returns
    -------
    (np.array, list of str, list of str)
       The first member is a dense 2d Numpy array, and the second
       and third are lists of strings (row names and column names,
       respectively). The third (column names) is None if the
       input file has no header. The row names are assumed always
       to be present in the leftmost column.
    """
    reader = csv.reader(open(src_filename), delimiter=delimiter, quoting=quoting)
    colnames = None
    if header:
        colnames = next(reader)
        colnames = colnames[1: ]
    mat = []
    rownames = []
    for line in reader:
        rownames.append(line[0])
        mat.append(np.array(list(map(float, line[1: ]))))
    return torch.FloatTensor(mat), rownames, colnames


def build_glove(src_filename):
    """Wrapper for using `build` to read in a GloVe file as a matrix"""
    return build(src_filename, delimiter=' ', header=False, quoting=csv.QUOTE_NONE)


def glove2dict(src_filename):
    """GloVe Reader.
    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.
    Returns
    -------
    dict
        Mapping words to their GloVe vectors.
    """
    reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE)
    return {line[0]: torch.FloatTensor(list(map(float, line[1: ]))) for line in reader}


def word_to_idx(word, glove_dict):
    """Maps a word to an idx in the GloVe dictionary.
    Parameters
    ----------
    word : str
        The word in question
    glove_dict : dict
        Dictionary containing str : FloatTensor for GloVe embedding vector
    Returns
    -------
    idx: int
        The integer index of the word in the glove vector
    """
    return glove_dict.get(word)[0]


def word_to_tensor(word, glove_dict):
    """Maps a word to a float tensor from the GloVe dictionary.
    Parameters
    ----------
    word : str
        The word in question
    glove_dict : dict
        Dictionary containing str : FloatTensor for GloVe embedding vector
    Returns
    -------
    tensor : torch.FloatTensor
        The integer index of the word in the glove vector
    """
    return glove_dict.get(word)[1]