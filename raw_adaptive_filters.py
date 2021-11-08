import numpy as np
from numba import jit, njit


@njit
def raw_ls(d, u, f_ord=12, f_delay=None, sigma=0., rcond=0.):
    """
    :param d: reference signal
    :param u: input signal
    :param f_ord: filter order
    :param f_delay: desired delay of filter
    :param sigma: regularization Rxx + (sigma**2)*np.eye(f_ord)
    :param rcond: singular values constraint like pinv
    :return: w - resulting filter coefficients, y - filter output
    """

    if f_delay is None:
        f_delay = f_ord//2 - 1

    u = np.pad(u, (f_ord - f_delay - 1, f_delay), mode='constant', constant_values=(0, 0))
    U = np.array([u[i:i+f_ord] for i in range(len(d))])
    Uh = U.conj().T

    # find filter weights by ls with regularization
    w = np.linalg.pinv((Uh @ U) + (sigma**2)*np.eye(f_ord), rcond=rcond) @ Uh @ d

    # filter output
    y = U @ w

    return w, y


@njit
def raw_lms(d, u, f_ord=12, f_delay=None, lr=None, w_init=None):

    if f_delay is None:
        f_delay = f_ord//2 - 1

    if w_init is None:
        w_init = np.random.rand(f_ord)/f_ord + 1j*np.random.rand(f_ord)/f_ord
    w = w_init.astype(complex)

    u = np.pad(u, (f_ord - f_delay - 1, f_delay), mode='constant', constant_values=(0, 0))
    y = np.zeros(len(d), dtype=complex)

    for i in range(len(d)):
        y[i] = u[i:i+f_ord] @ w
        err = d[i] - y[i]
        w = w + lr * err * u[i:i + f_ord].conj()

    return w, y


@njit
def raw_rls(d, u, f_ord=12, f_delay=None, sigma=10e3, w_init=None):

    if f_delay is None:
        f_delay = f_ord // 2 - 1

    if w_init is None:
        w_init = np.random.rand(f_ord)/f_ord + 1j*np.random.rand(f_ord)/f_ord
    w = w_init.astype(complex)

    u = np.pad(u, (f_ord - f_delay - 1, f_delay), mode='constant', constant_values=(0, 0))

    y = np.zeros(len(d), dtype=complex)
    P = sigma * np.eye(f_ord, dtype=complex)

    for i in range(len(d)):
        ui = u[i:i+f_ord]
        y[i] = ui @ w
        err = d[i] - y[i]
        K = (P @ ui.conj())/(1 + ui@P@ui.conj())
        P -= K.reshape((10, 1))@ui.reshape((1, 10))@P
        w += K*err

    return w, y
