import os
import ctypes as ct
from ctypes import c_int, c_double, POINTER
from typing import Optional, Tuple

import numpy as np
from functools import lru_cache


def _find_dll() -> Optional[str]:
    here = os.path.dirname(__file__)
    candidates = []
    # Windows typical build output
    candidates.append(os.path.join(here, 'native_dynamics', 'build', 'Release', 'native_dynamics.dll'))
    candidates.append(os.path.join(here, 'native_dynamics', 'build', 'Debug', 'native_dynamics.dll'))
    # Also allow same folder
    candidates.append(os.path.join(here, 'native_dynamics.dll'))
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Finally search PATH
    name = 'native_dynamics.dll'
    for path in os.getenv('PATH', '').split(os.pathsep):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            return p
    return None


class NativeDynamics:
    def __init__(self) -> None:
        dll_path = _find_dll()
        if not dll_path:
            raise FileNotFoundError('native_dynamics.dll not found. Build it under native_dynamics/')
        self._dll = ct.CDLL(dll_path)

        self._dll.dyn_compute_mf_batch.argtypes = [
            c_int,
            POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
            POINTER(c_double), POINTER(c_double),
        ]
        self._dll.dyn_compute_mf_batch.restype = c_int

        self._dll.dyn_compute_tau_chain.argtypes = [
            c_int,
            POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
            POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        ]
        self._dll.dyn_compute_tau_chain.restype = c_int

    def compute_mf_batch(self,
                          I_batch: np.ndarray,
                          m_batch: np.ndarray,
                          omega: np.ndarray,
                          dot_omega: np.ndarray,
                          ddpg: np.ndarray,
                          g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = int(I_batch.shape[0])
        assert I_batch.shape == (N, 3, 3)
        assert m_batch.shape == (N,), m_batch.shape
        assert omega.shape == (N, 3)
        assert dot_omega.shape == (N, 3)
        assert ddpg.shape == (N, 3)
        assert g.shape == (3,)

        I_c = np.ascontiguousarray(I_batch, dtype=np.float64)
        m_c = np.ascontiguousarray(m_batch, dtype=np.float64)
        w_c = np.ascontiguousarray(omega, dtype=np.float64)
        dw_c = np.ascontiguousarray(dot_omega, dtype=np.float64)
        a_c = np.ascontiguousarray(ddpg, dtype=np.float64)
        g_c = np.ascontiguousarray(g, dtype=np.float64)
        M_out = np.empty((N, 3), dtype=np.float64)
        F_out = np.empty((N, 3), dtype=np.float64)

        ret = self._dll.dyn_compute_mf_batch(
            N,
            I_c.ctypes.data_as(POINTER(c_double)),
            m_c.ctypes.data_as(POINTER(c_double)),
            w_c.ctypes.data_as(POINTER(c_double)),
            dw_c.ctypes.data_as(POINTER(c_double)),
            a_c.ctypes.data_as(POINTER(c_double)),
            g_c.ctypes.data_as(POINTER(c_double)),
            M_out.ctypes.data_as(POINTER(c_double)),
            F_out.ctypes.data_as(POINTER(c_double)),
        )
        if ret != 0:
            raise RuntimeError(f'dyn_compute_mf_batch failed: {ret}')
        return M_out, F_out

    def compute_tau_chain(self,
                          Ms: np.ndarray,
                          Fs: np.ndarray,
                          r_gs: np.ndarray,
                          p1s: np.ndarray,
                          tau_E: np.ndarray,
                          f_E: np.ndarray,
                          r_x: np.ndarray) -> np.ndarray:
        N = int(Ms.shape[0])
        assert Ms.shape == (N, 3)
        assert Fs.shape == (N, 3)
        assert r_gs.shape == (N, 3)
        assert p1s.shape == (N, 3)
        assert tau_E.shape == (3,)
        assert f_E.shape == (3,)
        assert r_x.shape == (3,)

        Ms_c = np.ascontiguousarray(Ms, dtype=np.float64)
        Fs_c = np.ascontiguousarray(Fs, dtype=np.float64)
        r_gs_c = np.ascontiguousarray(r_gs, dtype=np.float64)
        p1s_c = np.ascontiguousarray(p1s, dtype=np.float64)
        tau_E_c = np.ascontiguousarray(tau_E, dtype=np.float64)
        f_E_c = np.ascontiguousarray(f_E, dtype=np.float64)
        r_x_c = np.ascontiguousarray(r_x, dtype=np.float64)
        tau_out = np.empty((N, 3), dtype=np.float64)

        ret = self._dll.dyn_compute_tau_chain(
            N,
            Ms_c.ctypes.data_as(POINTER(c_double)),
            Fs_c.ctypes.data_as(POINTER(c_double)),
            r_gs_c.ctypes.data_as(POINTER(c_double)),
            p1s_c.ctypes.data_as(POINTER(c_double)),
            tau_E_c.ctypes.data_as(POINTER(c_double)),
            f_E_c.ctypes.data_as(POINTER(c_double)),
            r_x_c.ctypes.data_as(POINTER(c_double)),
            tau_out.ctypes.data_as(POINTER(c_double)),
        )
        if ret != 0:
            raise RuntimeError(f'dyn_compute_tau_chain failed: {ret}')
        return tau_out


@lru_cache(maxsize=1)
def get_native() -> Optional[NativeDynamics]:
    try:
        return NativeDynamics()
    except (OSError, FileNotFoundError):
        return None
