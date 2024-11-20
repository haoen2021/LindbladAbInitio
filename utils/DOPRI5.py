# RK_dopri 5 with adaptive step
import numpy as np
from copy import deepcopy
def DOPRI5(f, y0, t_initial, t_final, H_mat, K_matlist, h=0.01, atol=1e-6, rtol=1e-6):
    a = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
    b = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [1 / 5, 0, 0, 0, 0, 0, 0],
                  [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                  [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]])
    c = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    c_star = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])

    t, y = t_initial, deepcopy(y0+0j)
    t_end = t_final
    h_lst = []
    N = len(y0)

    if t_final < t_initial and h > 0:
        h = -h  # Set step size to negative for backward evolution
    while (h > 0 and t < t_end) or (h < 0 and t > t_end):
        if (h > 0 and t + h > t_end) or (h < 0 and t + h < t_end):
            h = t_end - t

        k = np.zeros((7, N, N), dtype=complex)

        for i in range(7):
            y_i = y.copy()
            for j in range(i):
                y_i += h * b[i, j] * k[j]
            k[i] = f(y_i, H_mat, K_matlist)      

        y_next = y.copy()
        for i in range(7):
            y_next += h * c[i] * k[i]

        y_star = y.copy()
        for i in range(7):
            y_star += h * c_star[i] * k[i]

        diff = np.abs(y_next-y_star)
        scale = atol + rtol * np.maximum(np.abs(y_next), np.abs(y_star))
        err = np.linalg.norm(diff/scale, ord=2)/np.sqrt(N)

        rho = (1/err)**0.2

        if err <= 1:
            h_lst.append(h)
            t = t + h
            y = y_next
        fac = min(2, max(0.8 * rho, 0.5))
        h = fac*h

    return y