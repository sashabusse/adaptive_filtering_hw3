import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import psd
mat = sio.loadmat('test1_25p0.mat')
x = mat['pdin']
y = mat['pdout']
tx_freq = mat['tx_freq']
rx_freq = mat['rx_freq']
x = np.reshape(x, x.size)
y = np.reshape(y, y.size)
tx_freq = float(np.reshape(tx_freq, 1))
rx_freq = float(np.reshape(rx_freq, 1))

dw = 2*np.pi*(tx_freq-rx_freq)
#y = y.real*np.exp(-1j*dw) + 1j*y.imag*np.exp(1j*dw)
print(y.mean())
y-=np.mean(y)

Fs = 245.76E6
(X, frq) = psd.psd_welch(x, 2048, signal.windows.blackmanharris(2048), 1024, Fs)
(Y, frq) = psd.psd_welch(y, 2048, signal.windows.blackmanharris(2048), 1024, Fs)
plt.plot(frq/1E6, X, frq/1E6, Y)
plt.grid('True')

plt.show()