import numpy as np


def izzo_algorithm(R1, R2, dt, mu, cw=1, N=0):
    """
    Solves Lambert's problem using Dario Izzo's algorithm.

    Inputs:
    -----------
    R1 : Position vector at departure [m]
    R2 : Position vector at arrival [m]
    dt : Time of flight [s]
    mu : Gravitational parameter [m^3/s^2]
    cw : Clockwise motion flag (1 for prograde/clockwise, 0 for retrograde/counter-clockwise)
    N : Number of full revolutions

    Outputs:
    --------
    V1 : Velocity vector(s) at departure [m/s]
    V2 : Velocity vector(s) at arrival [m/s]
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)

    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    c = np.linalg.norm(R2 - R1)
    s = (r1 + r2 + c) / 2.0

    i_r1 = R1 / r1
    i_r2 = R2 / r2
    i_h = np.cross(i_r1, i_r2)
    i_h = i_h / np.linalg.norm(i_h)

    lam = np.sqrt(1.0 - c / s)
    i_t1 = np.cross(i_h, i_r1)
    i_t2 = np.cross(i_h, i_r2)
    i_t1 = i_t1 / np.linalg.norm(i_t1)
    i_t2 = i_t2 / np.linalg.norm(i_t2)

    if i_h[2] < 0:
        lam = -lam
        i_t1 = -i_t1
        i_t2 = -i_t2

    if not cw:
        lam = -lam
        i_t1 = -i_t1
        i_t2 = -i_t2

    T = np.sqrt(2.0 * mu / s ** 3) * dt
    N_max = T / np.pi
    N_max = min(N, int(N_max))

    T00 = np.arccos(lam) + lam * np.sqrt(1.0 - lam ** 2)
    T0 = T00 + N_max * np.pi
    T1 = 2.0 / 3.0 * (1.0 - lam ** 3)

    if N_max > 0:
        if T < T0:
            it = 0
            err = 1.0
            T_min = T0
            x_old = 0.0
            x_new = 0.0
            while True:
                DT, DDT, DDDT = dt_dx(x_old, T_min, lam)
                if DT != 0.0:
                    x_new = x_old - DT * DDT / (DDT ** 2 - DT - DDDT / 2.0)
                err = abs(x_old - x_new)
                if err < 1e-13 or it > 12:
                    break
                T_min = x2tof(x_new, N_max, lam)
                x_old = x_new
                it += 1
            if T_min > T:
                N_max -= 1
        N_max = min(N, N_max)

    x_list = []
    iters = []

    if T >= 100.0:
        x0 = (100.0 - T) / (T - T00 + 4.0)
    elif T <= T1:
        x0 = T1 * (T1 - T) / (2.0 / 5.0 * (1.0 - lam ** 5) * T) + 1.0
    else:
        x0 = (T / T00) ** (0.69314718055994529 / np.log(T1 / T00)) - 1.0

    it_val, x_conv = householder(T, x0, 0, 1e-5, 15, lam)
    x_list.append(x_conv)
    iters.append(it_val)

    for i in range(1, N_max):
        tmp = ((i * np.pi + np.pi) / (8.0 * T)) ** (2.0 / 3.0)
        x_val1 = (tmp - 1.0) / (tmp + 1.0)
        it_val1, x_conv1 = householder(T, x_val1, i, 1e-8, 15, lam)
        x_list.append(x_conv1)
        iters.append(it_val1)

        tmp = ((8.0 * T) / (i * np.pi)) ** (2.0 / 3.0)
        x_val2 = (tmp - 1.0) / (tmp + 1.0)
        it_val2, x_conv2 = householder(T, x_val2, i, 1e-8, 15, lam)
        x_list.append(x_conv2)
        iters.append(it_val2)

    gamma = np.sqrt(mu * s / 2.0)
    rho_val = (r1 - r2) / c
    sigma = np.sqrt(1.0 - rho_val ** 2)

    V1_list = []
    V2_list = []

    for xi in x_list:
        y = np.sqrt(1.0 - lam ** 2 + lam ** 2 * xi ** 2)
        v_r1 = gamma * ((lam * y - xi) - rho_val * (lam * y + xi)) / r1
        v_r2 = -gamma * ((lam * y - xi) + rho_val * (lam * y + xi)) / r2
        v_t = gamma * sigma * (y + lam * xi)
        v_t1 = v_t / r1
        v_t2 = v_t / r2

        v1_vec = v_r1 * i_r1 + v_t1 * i_t1
        v2_vec = v_r2 * i_r2 + v_t2 * i_t2
        V1_list.append(v1_vec)
        V2_list.append(v2_vec)

    if len(V1_list) == 1:
        return V1_list[0], V2_list[0]
    else:
        return np.array(V1_list), np.array(V2_list)


def householder(T, x0, N, eps, iter_max, lam):
    it = 0
    err = 1.0
    while (err > eps) and (it < iter_max):
        tof = x2tof(x0, N, lam)
        DT, DDT, DDDT = dt_dx(x0, tof, lam)
        delta = tof - T
        DT2 = DT * DT
        x_new = x0 - delta * (DT2 - delta * DDT / 2.0) / (DT * (DT2 - delta * DDT) + DDDT * delta ** 2 / 6.0)
        err = abs(x0 - x_new)
        x0 = x_new
        it += 1
    return it, x0


def dt_dx(x, T, lam):
    y = np.sqrt(1.0 - lam ** 2 * (1.0 - x ** 2))
    DT = 1.0 / (1.0 - x ** 2) * (3.0 * T * x - 2.0 * lam ** 3 * x / y)
    DDT = 1.0 / (1.0 - x ** 2) * (3.0 * T + 5.0 * x * DT + 2.0 * (1.0 - lam ** 2) * lam ** 3 / y ** 3)
    DDDT = 1.0 / (1.0 - x ** 2) * (7.0 * x * DDT + 8.0 * DT - 6.0 * (1.0 - lam ** 2) * lam ** 5 * x / y ** 5)
    return DT, DDT, DDDT


def x2tof2(x, N, lam):
    a = 1.0 / (1.0 - x ** 2)
    if a > 0:
        alfa = 2.0 * np.arccos(x)
        beta = 2.0 * np.arcsin(np.sqrt(lam ** 2 / a))
        if lam < 0:
            beta = -beta
        tof = (a ** 1.5 * ((alfa - np.sin(alfa)) - (beta - np.sin(beta)) + 2.0 * np.pi * N)) / 2.0
    else:
        alfa = 2.0 * np.arccosh(x)
        beta = 2.0 * np.arcsinh(np.sqrt(-lam ** 2 / a))
        if lam < 0:
            beta = -beta
        tof = -a * np.sqrt(-a) * ((beta - np.sinh(beta)) - (alfa - np.sinh(alfa))) / 2.0
    return tof


def x2tof(x, N, lam):
    battin = 0.01
    lagrange = 0.2
    dist = abs(x - 1.0)
    if dist < lagrange and dist > battin:
        return x2tof2(x, N, lam)

    K = lam ** 2
    E = x ** 2 - 1.0
    rho = abs(E)
    z = np.sqrt(1.0 + K * E)
    if dist < battin:
        eta = z - lam * x
        S1 = 0.5 * (1.0 - lam - x * eta)
        Q = 4.0 / 3.0 * hypergeometric_f(S1, 1e-11)
        tof = (eta ** 3 * Q + 4.0 * lam * eta) / 2.0 + N * np.pi / rho ** 1.5
    else:
        y = np.sqrt(rho)
        g = x * z - lam * E
        if E < 0:
            l = np.arccos(g)
            d = N * np.pi + l
        else:
            f = y * (z - lam * x)
            d = np.log(f + g)
        tof = (x - lam * z - d / y) / E
    return tof


def hypergeometric_f(z, tol):
    Sj = 1.0
    Cj = 1.0
    err = 1.0
    j = 0
    while err > tol:
        Cj1 = Cj * (3 + j) * (1 + j) / (2.5 + j) * z / (j + 1)
        Sj1 = Sj + Cj1
        err = abs(Cj1)
        Sj = Sj1
        Cj = Cj1
        j += 1
    return Sj