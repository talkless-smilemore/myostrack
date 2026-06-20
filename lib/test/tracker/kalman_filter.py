"""
Adaptive Kalman Filter for Anti-UAV tracking.

Implements a Constant Velocity (CV) Kalman filter with innovation-based
process noise adaptation.  When the Mahalanobis distance of the observation
innovation exceeds a threshold the process noise Q is temporarily increased,
allowing the filter to respond quickly to drone manoeuvres. During steady
flight Q decays back toward its baseline.

State:  x = [cx, cy, w, h, vx, vy]^T      (6,)
Obs:    z = [cx, cy, w, h]^T               (4,)
"""

import math
import torch


class AdaptiveKalmanFilter:
    def __init__(
        self,
        dt: float = 1.0,
        base_process_noise: float = 0.01,
        velocity_noise_scale: float = 10.0,
        size_noise_scale: float = 2.0,
        base_observation_noise: float = 5.0,
        adapt_threshold: float = 5.0,
        adapt_gain: float = 2.0,
        decay_rate: float = 0.95,
        max_q_scale: float = 10.0,
    ):
        self.dt = dt

        # ---- state transition F (6x6) ----
        self.F = torch.eye(6, dtype=torch.float32)
        self.F[0, 4] = dt
        self.F[1, 5] = dt

        # ---- observation matrix H (4x6) ----
        self.H = torch.zeros(4, 6, dtype=torch.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # ---- base process noise Q ----
        self._Q_base = torch.diag(
            torch.tensor(
                [
                    base_process_noise,
                    base_process_noise,
                    base_process_noise * size_noise_scale,
                    base_process_noise * size_noise_scale,
                    base_process_noise * velocity_noise_scale,
                    base_process_noise * velocity_noise_scale,
                ],
                dtype=torch.float32,
            )
        )
        self.Q = self._Q_base.clone()
        self._q_scale = 1.0

        # ---- observation noise R ----
        self.R = torch.diag(
            torch.tensor(
                [
                    base_observation_noise,
                    base_observation_noise,
                    base_observation_noise,
                    base_observation_noise,
                ],
                dtype=torch.float32,
            )
        )

        # ---- adaptation parameters ----
        self.adapt_threshold = adapt_threshold
        self.adapt_gain = adapt_gain
        self.decay_rate = decay_rate
        self.max_q_scale = max_q_scale

        # ---- filter state ----
        self.x = None  # (6,)
        self.P = None  # (6,6)

        # ---- diagnostics ----
        self._last_mahalanobis = 0.0

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def init(self, cx: float, cy: float, w: float, h: float):
        """Initialise filter from the first-frame bounding box."""
        self.x = torch.tensor(
            [cx, cy, w, h, 0.0, 0.0], dtype=torch.float32
        )
        self.P = torch.eye(6, dtype=torch.float32) * 100.0
        self._q_scale = 1.0
        self.Q = self._Q_base.clone()

    def predict(self):
        """Prior prediction. Returns predicted [cx, cy, w, h]."""
        if self.x is None:
            raise RuntimeError("KalmanFilter not initialised. Call init() first.")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].clone()

    def update(self, z_cx: float, z_cy: float, z_w: float, z_h: float):
        """Update with model observation.

        Returns the filtered state [cx, cy, w, h].
        """
        z = torch.tensor([z_cx, z_cy, z_w, z_h], dtype=torch.float32)
        y = z - self.H @ self.x           # innovation  (4,)
        S = self.H @ self.P @ self.H.T + self.R  # innovation cov

        K = self.P @ self.H.T @ torch.linalg.inv(S)   # Kalman gain
        self.x = self.x + K @ y
        self.P = (torch.eye(6) - K @ self.H) @ self.P

        self._adapt_q(y, S)
        return self.x[:4].clone()

    def update_with_confidence(
        self, z_cx, z_cy, z_w, z_h, confidence: float, conf_threshold: float = 0.3
    ):
        """Confidence-gated update.

        Returns (filtered_box, used_observation).
        """
        if confidence > conf_threshold:
            return self.update(z_cx, z_cy, z_w, z_h), True
        return self.x[:4].clone(), False

    def get_state(self):
        if self.x is None:
            raise RuntimeError("KalmanFilter not initialised.")
        return self.x[:4].clone()

    def get_velocity(self):
        if self.x is None:
            return torch.tensor([0.0, 0.0])
        return self.x[4:6].clone()

    @property
    def estimated_speed(self) -> float:
        vx, vy = self.get_velocity()
        return float(math.sqrt(vx.item() ** 2 + vy.item() ** 2))

    # -----------------------------------------------------------------
    # adaptive Q
    # -----------------------------------------------------------------

    def _adapt_q(self, innovation, innovation_cov):
        d2 = float(innovation @ torch.linalg.inv(innovation_cov) @ innovation)
        self._last_mahalanobis = math.sqrt(max(0.0, d2))

        if self._last_mahalanobis > self.adapt_threshold:
            self._q_scale = min(
                self._q_scale * self.adapt_gain, self.max_q_scale
            )
        else:
            self._q_scale = max(self._q_scale * self.decay_rate, 1.0)

        self.Q = self._Q_base * self._q_scale
