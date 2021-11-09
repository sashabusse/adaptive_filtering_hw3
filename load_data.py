import numpy as np
from scipy import io


def load_data(Fs, fname='data/test2_25p0.mat'):
    """
    reads file and makes all the corrections from hw2

    :param Fs: sampling frequency
    :param fname: file name to load data from
    :return: x - system input sequence, y - system output sequence (with all the corrections)
    """
    # << load data >> -------------------------------------------------------
    mat = io.loadmat(fname)
    x = np.reshape(mat['pdin'], (-1,))
    y = np.reshape(mat['pdout'], (-1,))
    tx_freq = float(mat['tx_freq'])
    rx_freq = float(mat['rx_freq'])
    # ------------------------------------------------------------------------

    # << compensate frequency shift and mean value >> --------------------------
    y = (y - np.mean(y)) * np.exp(-1j * 2 * np.pi * ((tx_freq - rx_freq) / Fs) * np.arange(len(y)))
    # --------------------------------------------------------------------------

    # << cut out piece of output that corresponds to the input >> -------------
    xy_corr = np.abs(np.correlate(y, x, mode='valid')) / len(x)
    st_ind = np.argmax(xy_corr)
    y = y[st_ind: st_ind + len(x)]
    # -------------------------------------------------------------------------

    # << gain control calculation >> ---------------------------------------
    g = y.conj().dot(x) / (y.conj().dot(y))
    y = y * g
    # -------------------------------------------------------------------------

    return x, y

