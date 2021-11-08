from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as signal


# << shortcut function for spd with given parameters >>------------------------------
def spd_welch_db(sig, fs, window=signal.windows.blackmanharris(2048),
                nperseg=2048, return_onesided=False, detrend=False):
    frq, psd = signal.welch(
                sig, fs=fs, window=window,
                nperseg=nperseg, return_onesided=return_onesided, detrend=detrend)
    psd = 10 * np.log10(np.fft.fftshift(psd))
    frq = np.fft.fftshift(frq)
    return frq, psd
# -------------------------------------------------------------------------------------


def plot_afc_db(sig, fs, ax):
    sig_fft = np.fft.fftshift(np.fft.fft(sig, n=1024))
    nus = np.arange(-len(sig_fft) // 2, len(sig_fft) // 2) / len(sig_fft)
    freqs = nus*fs/1e6
    ax.plot(freqs, 20*np.log10(np.abs(sig_fft)))


# << Utility functions to plot in one line >> -------------------------------
def plot_signal(x, y, Fs, suptitle, scale_pts=50, fontsize=15):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(suptitle, fontsize=18)

    ax[0].plot(np.arange(len(x)) * 1e6 / Fs, np.abs(x))
    ax[0].plot(np.arange(len(y)) * 1e6 / Fs, np.abs(y))
    ax[0].set_ylabel("модуль сигнала", fontsize=fontsize)
    ax[0].set_xlabel("время мкс", fontsize=fontsize)
    ax[0].legend(["abs(x)", "abs(y)"], fontsize=fontsize)
    ax[0].grid(True)

    ax[1].plot(np.arange(scale_pts) * 1e6 / Fs, np.abs(x[0:scale_pts]))
    ax[1].plot(np.arange(scale_pts) * 1e6 / Fs, np.abs(y[0:scale_pts]))
    ax[1].set_ylabel("модуль сигнала", fontsize=fontsize)
    ax[1].set_xlabel("время мкс", fontsize=fontsize)
    ax[1].legend(["abs(x)", "abs(y)"], fontsize=fontsize)
    ax[1].grid(True)


def plot_spd(signals, Fs, title, legend=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for signal in signals:
        frq, S = spd_welch_db(signal, fs=Fs)
        ax.plot(frq/1e6, S)

    ax.set_title(title)
    ax.set_xlabel("частота МГц")
    ax.set_ylabel("СПМ")

    if not(legend is None):
        ax. legend(legend)

    ax.grid(True)
# -------------------------------------------------------------------------------------
