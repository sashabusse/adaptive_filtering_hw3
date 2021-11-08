import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from Utility import plot_signal, plot_spd, plot_afc_db
from scipy import signal

# << load data >> -----------------------------------------------------------
Fs = 245.76E6
x, y = load_data(Fs, fname='data/test4_25p0.mat')
# ---------------------------------------------------------------------------

# << plot input for this stage >> -------------------------------------------------
plot_spd([x, y, x-y], Fs, title="входной сигнал", legend=["x", "y", "err"])
# ---------------------------------------------------------------------------





LMS_meth(x, y, Fs)
