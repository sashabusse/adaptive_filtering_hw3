from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as signal


def spd_welch_db(sig, fs, window=signal.windows.blackmanharris(2048),
                nperseg=2048, return_onesided=False, detrend=False):
    frq, psd = signal.welch(
                sig, fs=fs, window=window,
                nperseg=nperseg, return_onesided=return_onesided, detrend=detrend)
    psd = 10 * np.log10(np.fft.fftshift(psd))
    frq = np.fft.fftshift(frq)
    return frq, psd


def plot_afc_db(irs, fs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_xlabel("частота МГц")
    for ir in irs:
        sig_fft = np.fft.fftshift(np.fft.fft(ir, n=1024))
        freq = (np.arange(-len(sig_fft) // 2, len(sig_fft) // 2) / len(sig_fft)) * fs/1e6
        ax.plot(freq, 20*np.log10(np.abs(sig_fft)))


def plot_spd(signals, Fs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for s in signals:
        frq, S = spd_welch_db(s, fs=Fs)
        ax.plot(frq/1e6, S)

    ax.set_xlabel("частота МГц")
    ax.grid(True)

