# CREDIT GOES TO CHRISTOPHER POTTS FOR THIS CODE

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2016"

import sys
import csv
import numpy as np
import torch

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
    return (torch.FloatTensor(mat), rownames, colnames)


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


def d_tanh(z):
    """The derivative of np.tanh. z should be a float or np.array."""
    return 1.0 - z**2

def softmax(z):
    """Softmax activation function. z should be a float or np.array."""
    # Increases numerical stability:
    t = np.exp(z - torch.max(z))
    return t / torch.sum(t)

def progress_bar(msg):
    """Simple over-writing progress bar."""
    sys.stderr.write('\r')
    sys.stderr.write(msg)
    sys.stderr.flush()