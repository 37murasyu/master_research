from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class KalmanFilter1D:
    """
    1次元（位置・速度・加速度）カルマンフィルタ。

    状態は ``[position, velocity, acceleration]``、観測は位置のみ。
    ``predict(dt)`` ごとに ``F`` と ``Q`` を ``dt`` から再計算する。
    """

    def __init__(
        self,
        process_noise_intensity: float,
        measurement_noise_variance: float,
        initial_position: float,
        initial_velocity: float = 0.0,
        initial_acceleration: float = 0.0,
        initial_position_variance: float = 1.0,
        initial_velocity_variance: float = 1.0,
        initial_acceleration_variance: float = 1.0,
    ) -> None:
        self.q = float(process_noise_intensity)
        self.R = float(measurement_noise_variance)
        self.H = np.array([[1.0, 0.0, 0.0]], dtype=float)

        self._x = np.array(
            [initial_position, initial_velocity, initial_acceleration],
            dtype=float,
        )
        self._P = np.diag(
            [
                float(initial_position_variance),
                float(initial_velocity_variance),
                float(initial_acceleration_variance),
            ]
        )

        self._F = np.eye(3, dtype=float)
        self._Q = np.zeros((3, 3), dtype=float)

    def _compute_transition_and_process_noise(self, dt: float) -> None:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        dt5 = dt4 * dt

        self._F = np.array(
            [
                [1.0, dt, 0.5 * dt2],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        self._Q = self.q * np.array(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
                [dt3 / 6.0, dt2 / 2.0, dt],
            ],
            dtype=float,
        )

    def predict(self, dt: float) -> None:
        """
        時間更新ステップ。``dt`` から ``F`` と ``Q`` を再計算し、状態を予測する。
        """

        if dt <= 0:
            raise ValueError("dt must be positive")

        self._compute_transition_and_process_noise(dt)
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def update(self, measurement: float | None) -> None:
        """
        観測更新ステップ。観測が ``None`` のときは何もしない。
        """

        if measurement is None:
            return

        z = float(measurement)
        y = z - float(self.H @ self._x)  # innovation

        S = float(self.H @ self._P @ self.H.T + self.R)  # innovation covariance (scalar)
        if S <= 0:
            raise ValueError("Innovation covariance must be positive")

        K = (self._P @ self.H.T) / S  # Kalman gain (3x1)

        self._x = self._x + (K[:, 0] * y)

        I = np.eye(3, dtype=float)
        KH = K @ self.H
        # Joseph form to keep P symmetric / positive semi-definite
        self._P = (I - KH) @ self._P @ (I - KH).T + K * self.R * K.T

    @property
    def state(self) -> Tuple[float, float, float]:
        return tuple(self._x.tolist())  # (position, velocity, acceleration)

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()


class KalmanFilterND:
    """
    同一モデルを複数軸に独立適用するラッパ（例: 3D xyz）。
    """

    def __init__(
        self,
        process_noise_intensity: float,
        measurement_noise_variance: float,
        initial_position: Sequence[float],
        initial_velocity: Sequence[float] | float = 0.0,
        initial_acceleration: Sequence[float] | float = 0.0,
        initial_position_variance: Sequence[float] | float = 1.0,
        initial_velocity_variance: Sequence[float] | float = 1.0,
        initial_acceleration_variance: Sequence[float] | float = 1.0,
    ) -> None:
        positions = np.asarray(initial_position, dtype=float)
        n = positions.shape[0]

        def _as_array(val: Sequence[float] | float) -> np.ndarray:
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 0:
                arr = np.full(n, float(arr))
            if arr.shape[0] != n:
                raise ValueError("Initial value length mismatch")
            return arr

        velocities = _as_array(initial_velocity)
        accelerations = _as_array(initial_acceleration)
        pos_vars = _as_array(initial_position_variance)
        vel_vars = _as_array(initial_velocity_variance)
        acc_vars = _as_array(initial_acceleration_variance)

        self.filters: list[KalmanFilter1D] = []
        for i in range(n):
            self.filters.append(
                KalmanFilter1D(
                    process_noise_intensity=process_noise_intensity,
                    measurement_noise_variance=measurement_noise_variance,
                    initial_position=positions[i],
                    initial_velocity=velocities[i],
                    initial_acceleration=accelerations[i],
                    initial_position_variance=pos_vars[i],
                    initial_velocity_variance=vel_vars[i],
                    initial_acceleration_variance=acc_vars[i],
                )
            )

    def predict(self, dt: float) -> None:
        for f in self.filters:
            f.predict(dt)

    def update(self, measurement: Sequence[float] | None) -> None:
        if measurement is None:
            return
        meas_arr = np.asarray(measurement, dtype=float)
        if meas_arr.shape[0] != len(self.filters):
            raise ValueError("Measurement dimension mismatch")
        for m, f in zip(meas_arr, self.filters):
            f.update(float(m))

    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.array([f.state[0] for f in self.filters], dtype=float)
        velocities = np.array([f.state[1] for f in self.filters], dtype=float)
        accelerations = np.array([f.state[2] for f in self.filters], dtype=float)
        return positions, velocities, accelerations

    @property
    def covariance(self) -> np.ndarray:
        return np.stack([f.covariance for f in self.filters], axis=0)


__all__ = ["KalmanFilter1D", "KalmanFilterND"]
