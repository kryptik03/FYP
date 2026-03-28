import numpy as np

c = 3e8


def func_PD_Simulation(receiver_A, receiver_B, receiver_C, receiver_D, PD_coords):
    """
    All function inputs are 3D coordinates (3 x 1 column vectors / length-3 arrays)
    Returns the TDOA between receivers AB, AC and AD
    positive t_AB corresponds to greater distance to B than to A
    """
    t_AB = (np.linalg.norm(receiver_B - PD_coords) - np.linalg.norm(receiver_A - PD_coords)) / c
    t_AC = (np.linalg.norm(receiver_C - PD_coords) - np.linalg.norm(receiver_A - PD_coords)) / c
    t_AD = (np.linalg.norm(receiver_D - PD_coords) - np.linalg.norm(receiver_A - PD_coords)) / c
    return t_AB, t_AC, t_AD


if __name__ == "__main__":
    rcv_A = np.array([-0.9, -0.9, 1.0])
    rcv_B = np.array([0.9, -0.9, 0.0])
    rcv_C = np.array([-0.9, 0.9, 0.0])
    rcv_D = np.array([0.9, 0.9, 0.0])
    PD = np.array([2.1, 7.0, -5.0])

    t_AB, t_AC, t_AD = func_PD_Simulation(rcv_A, rcv_B, rcv_C, rcv_D, PD)

    d = 0.01  # delta value for distance
    d_t = 1e-10  # delta value for time

    def f1(x, y, z, t):
        return np.linalg.norm(np.array([x - rcv_A[0], y - rcv_A[1], z - rcv_A[2]])) - c * t

    def f2(x, y, z, t):
        return np.linalg.norm(np.array([x - rcv_B[0], y - rcv_B[1], z - rcv_B[2]])) - c * (t + t_AB)

    def f3(x, y, z, t):
        return np.linalg.norm(np.array([x - rcv_C[0], y - rcv_C[1], z - rcv_C[2]])) - c * (t + t_AC)

    def f4(x, y, z, t):
        return np.linalg.norm(np.array([x - rcv_D[0], y - rcv_D[1], z - rcv_D[2]])) - c * (t + t_AD)

    def J(x, y, z, t):
        # finite-difference Jacobian same ordering as MATLAB: [df/dx df/dy df/dz df/dt] per row
        return np.array([
            [(f1(x + d, y, z, t) - f1(x, y, z, t)) / d, (f1(x, y + d, z, t) - f1(x, y, z, t)) / d,
             (f1(x, y, z + d, t) - f1(x, y, z, t)) / d, (f1(x, y, z, t + d_t) - f1(x, y, z, t)) / d_t],
            [(f2(x + d, y, z, t) - f2(x, y, z, t)) / d, (f2(x, y + d, z, t) - f2(x, y, z, t)) / d,
             (f2(x, y, z + d, t) - f2(x, y, z, t)) / d, (f2(x, y, z, t + d_t) - f2(x, y, z, t)) / d_t],
            [(f3(x + d, y, z, t) - f3(x, y, z, t)) / d, (f3(x, y + d, z, t) - f3(x, y, z, t)) / d,
             (f3(x, y, z + d, t) - f3(x, y, z, t)) / d, (f3(x, y, z, t + d_t) - f3(x, y, z, t)) / d_t],
            [(f4(x + d, y, z, t) - f4(x, y, z, t)) / d, (f4(x, y + d, z, t) - f4(x, y, z, t)) / d,
             (f4(x, y, z + d, t) - f4(x, y, z, t)) / d, (f4(x, y, z, t + d_t) - f4(x, y, z, t)) / d_t]
        ])

    r = np.array([0.0, 0.0, 0.0, 0.0])  # initial result
    Na = 0.0001
    check = abs(f1(r[0], r[1], r[2], r[3])) + abs(f2(r[0], r[1], r[2], r[3])) + abs(f3(r[0], r[1], r[2], r[3])) + abs(
        f4(r[0], r[1], r[2], r[3]))
    prm = 0.0  # parametric variable used to generate coordinates along a multidimensional spiral

    while (abs(check) > Na) and (prm <= 100):
        r[0] = prm * np.cos(prm)
        r[1] = prm * np.sin(prm)
        r[2] = prm * np.cos(prm)
        r[3] = (prm * np.sin(prm)) / 1e8
        check = np.inf

        iter_modsec = 0  # mod secant iterator

        while (abs(check) > Na) and (iter_modsec <= 10000):
            r_0 = r.copy()
            J_0 = J(r_0[0], r_0[1], r_0[2], r_0[3])
            f_0 = np.array([f1(r_0[0], r_0[1], r_0[2], r_0[3]), f2(r_0[0], r_0[1], r_0[2], r_0[3]),
                            f3(r_0[0], r_0[1], r_0[2], r_0[3]), f4(r_0[0], r_0[1], r_0[2], r_0[3])])

            # Solve linear system J_0 * delta = f_0  -> delta = J_0 \ f_0 in MATLAB
            # MATLAB does r = r_0 - J_0\f_0
            # Use numpy.linalg.solve with fallback to lstsq when singular
            try:
                delta = np.linalg.solve(J_0, f_0)
            except np.linalg.LinAlgError:
                # if J_0 is singular, skip to the next prm value
                break

            r = r_0 - delta

            check = abs(f1(r[0], r[1], r[2], r[3])) + abs(f2(r[0], r[1], r[2], r[3])) + abs(
                f3(r[0], r[1], r[2], r[3])) + abs(f4(r[0], r[1], r[2], r[3]))
            iter_modsec = iter_modsec + 1
        
        prm = prm + 0.1

    print("t_AB, t_AC, t_AD =", t_AB, t_AC, t_AD)
    print("solution r =", r)
    print("check =", check)
