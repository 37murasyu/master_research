import numpy as np

from utils_dynamic import compute_MF_batch_native, compute_tau_chain_native


def test_mf_small():
    N = 2
    I_batch = np.array([
        np.eye(3),
        np.diag([2.0, 3.0, 4.0])
    ], dtype=np.float64)
    m_batch = np.array([1.5, 2.0], dtype=np.float64)
    w = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    dw = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    a = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float64)
    g = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    M, F = compute_MF_batch_native(I_batch, m_batch, w, dw, a, g)

    # numpy reference
    M_ref = np.empty((N,3))
    F_ref = np.empty((N,3))
    for i in range(N):
        Iw = I_batch[i] @ w[i]
        M_ref[i] = I_batch[i] @ dw[i] + np.cross(w[i], Iw)
        F_ref[i] = m_batch[i] * (a[i] - g)

    assert np.allclose(M, M_ref)
    assert np.allclose(F, F_ref)


def test_tau_small():
    Ms = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float64)
    Fs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    r_gs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    p1s = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    tau_E = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    f_E = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    r_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tau = compute_tau_chain_native(Ms, Fs, r_gs, p1s, tau_E, f_E, r_x)

    # naive numpy
    tau_ref = []
    N = Ms.shape[0]
    for j in range(N):
        t = np.sum(Ms[j:], axis=0)
        for i in range(j, N):
            t = t + np.cross(r_gs[i] - p1s[j], Fs[i])
        t = t - tau_E - np.cross(r_x - p1s[j], f_E)
        tau_ref.append(t)
    tau_ref = np.vstack(tau_ref)

    assert np.allclose(tau, tau_ref)


if __name__ == '__main__':
    test_mf_small()
    test_tau_small()
    print('native_dynamics tests passed (native or numpy fallback).')
