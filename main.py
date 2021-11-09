import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from load_data import load_data
from Utility import plot_spd, plot_afc_db
from raw_adaptive_filters import raw_ls, raw_rls, raw_lms

# << load data >> -----------------------------------------------------------
Fs = 245.76E6
d, u = load_data(Fs, fname='data/test4_25p0.mat')
# ---------------------------------------------------------------------------

# << plot input for this stage >> -------------------------------------------------
fig_in, ax_in = plt.subplots(1, 1)
plot_spd([d, u], Fs, ax_in)
ax_in.set_ylabel("dB")
ax_in.legend(["d", "u", "err (d-u)"])
ax_in.set_title("input signal")
# ---------------------------------------------------------------------------

# << auto correlation of input >> ------------------------------------------
fig_ac, ax_ac = plt.subplots(1, 1)
rdu = np.correlate(d, u, mode="same")
rdu_x = np.arange(len(rdu)) - len(rdu)//2
ax_ac.plot(rdu_x, np.abs(rdu)/np.max(np.abs(rdu)))
ax_ac.set_title("cross correlation (d, u)")
ax_ac.set_xlabel("смещение в отсчетах")
ax_ac.grid(True)
# ---------------------------------------------------------------------------


# << compare 3 and 20 order of filter >> ----------------------------------------------------
compare_3_and_20 = False
if compare_3_and_20:
    # regularization not needed but still let it be less then u noise level
    reg_sigma = np.sqrt((10**(-4))*Fs)
    w_ls3, y_ls3 = raw_ls(d, u, ir_len=4, f_delay=1, sigma=reg_sigma, rcond=0.)
    w_ls20, y_ls20 = raw_ls(d, u, ir_len=21, f_delay=None, sigma=reg_sigma, rcond=0.)
    fig_ls, ax_ls = plt.subplots(1, 1)
    plot_spd([d, y_ls3, d-y_ls3, y_ls20, d-y_ls20], Fs, ax=ax_ls)
    plot_afc_db([w_ls3], Fs, ax=ax_ls)
    plot_afc_db([w_ls20], Fs, ax=ax_ls)
    ax_ls.set_ylabel("dB")
    ax_ls.legend(["d", "y order=3", "err order=3", "y order=20", "err order=20", "afc order=3", "afc order=20"])
    ax_ls.set_title("ls optimization results")
# ---------------------------------------------------------------------------


# << compare 2, 3 order of filter >> ----------------------------------------------------
compare_2_3 = False
if compare_2_3:
    reg_sigma = np.sqrt((10**(-4))*Fs)
    w_ls2, y_ls2 = raw_ls(d, u, ir_len=3, f_delay=0, sigma=reg_sigma, rcond=0.)
    w_ls3, y_ls3 = raw_ls(d, u, ir_len=4, f_delay=0, sigma=reg_sigma, rcond=0.)
    fig_ls, ax_ls = plt.subplots(1, 1)
    plot_spd([d, y_ls2, d-y_ls2, y_ls3, d-y_ls3], Fs, ax=ax_ls)
    plot_afc_db([w_ls2], Fs, ax=ax_ls)
    plot_afc_db([w_ls3], Fs, ax=ax_ls)
    ax_ls.set_ylabel("dB")
    ax_ls.legend(["d", "y order=2", "err order=2", "y order=3", "err order=3", "afc order=2", "afc order=3"])
    ax_ls.set_title("ls optimization results")

# ---------------------------------------------------------------------------


# << compare 2, 3 order of filter >> ----------------------------------------------------
compare_2_1 = False
if compare_2_1:
    reg_sigma = np.sqrt((10**(-4))*Fs)
    w_ls2, y_ls2 = raw_ls(d, u, ir_len=3, f_delay=0, sigma=reg_sigma, rcond=0.)
    w_ls1, y_ls1 = raw_ls(d, u, ir_len=2, f_delay=0, sigma=reg_sigma, rcond=0.)
    fig_ls, ax_ls = plt.subplots(1, 1)
    plot_spd([d, y_ls2, d-y_ls2, y_ls1, d-y_ls1], Fs, ax=ax_ls)
    plot_afc_db([w_ls2], Fs, ax=ax_ls)
    plot_afc_db([w_ls1], Fs, ax=ax_ls)
    ax_ls.set_ylabel("dB")
    ax_ls.legend(["d", "y order=2", "err order=2", "y order=1", "err order=1", "afc order=2", "afc order=1"])
    ax_ls.set_title("ls optimization results")
# ---------------------------------------------------------------------------

# << compare different delay >> ----------------------------------------------------
compare_delay = False
if compare_delay:
    reg_sigma = np.sqrt((10**(-4))*Fs)
    w_ls_del0, y_ls_del0 = raw_ls(d, u, ir_len=3, f_delay=0, sigma=reg_sigma, rcond=0.)
    w_ls_del1, y_ls_del1 = raw_ls(d, u, ir_len=3, f_delay=1, sigma=reg_sigma, rcond=0.)
    w_ls_del2, y_ls_del2 = raw_ls(d, u, ir_len=3, f_delay=2, sigma=reg_sigma, rcond=0.)

    fig_ls, ax_ls = plt.subplots(1, 1)
    plot_spd([d, d-y_ls_del0, d-y_ls_del1], Fs, ax=ax_ls)
    ax_ls.set_ylabel("dB")
    ax_ls.legend(["d", "err delay=0", "err delay=1"])
    ax_ls.set_title("ls optimization results (filter order = 2)")
# ---------------------------------------------------------------------------

# << adapt RLS >> -----------------------------------------------------------
w_rls, y_rls = raw_rls(d, u, ir_len=3, f_delay=1, sigma_init=rdu[0])
reg_sigma = np.sqrt((10**(-4))*Fs)
w_ls, y_ls = raw_ls(d, u, ir_len=3, f_delay=1, sigma=reg_sigma, rcond=0.)

fig_rls, ax_rls = plt.subplots(1, 1)
plot_spd([d, d-y_rls, d-y_ls], Fs, ax=ax_rls)
plot_afc_db([w_rls], Fs, ax=ax_rls)
plot_afc_db([w_ls], Fs, ax=ax_rls)

ax_rls.set_ylabel("dB")
ax_rls.legend(["d", "err rls", "err ls", "afc rls", "afc ls"])
ax_rls.set_title("ls optimization results (filter order=2, filter delay=1)")

fig_rls_err, ax_rls_err = plt.subplots(1, 1)
ax_rls_err.plot(signal.lfilter(np.ones(100)/100, 1., np.abs(d-y_rls))[100:])
ax_rls_err.set_xlabel("шаг адаптации")
ax_rls_err.set_ylabel("err_abs = np.abs(d-y_rls)")
ax_rls_err.grid(True)

fig_ir, ax_ir = plt.subplots(1, 1)
ax_ir.plot(w_rls.real)
ax_ir.plot(w_rls.imag)
ax_ir.legend(["real part", "imag part"])
ax_ir.set_xlabel("N отсчета импульсной характеристики")
ax_ir.set_title("импульсная характеристика RLS")
# ---------------------------------------------------------------------------
plt.show()

