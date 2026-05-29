import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def _awgn_noise(n: int, snr_lin: float, sig_pwr: float = 1.0) -> np.ndarray:
    sigma = np.sqrt(sig_pwr / (2.0 * snr_lin))
    return sigma * (np.random.randn(n) + 1j * np.random.randn(n))

class CompositeChannelModel:
    """Композитный канал: Fading -> AWGN -> Impulse. Нормировка E[|h|^2]=1."""
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rayleigh_en = cfg.get("rayleigh", {}).get("enabled", False)
        self.rician_en   = cfg.get("rician", {}).get("enabled", False)
        self.imp_en      = cfg.get("impulse_noise", {}).get("enabled", False)

        if self.rician_en: self.rayleigh_en = False

        if self.rayleigh_en:
            self.n_rays = int(cfg["rayleigh"].get("n_rays", 16))
            self.norm_dop = float(cfg["rayleigh"].get("normalized_doppler", 0.01))
        elif self.rician_en:
            self.k_lin = 10 ** (float(cfg["rician"].get("k_factor_dB", 3.0)) / 10.0)
            self.n_rays = int(cfg["rician"].get("n_rays", 16))
            self.norm_dop = float(cfg["rician"].get("normalized_doppler", 0.01))

        if self.imp_en:
            imp = cfg["impulse_noise"]
            self.imp_prob = float(imp.get("impulse_probability", 0.001))
            self.imp_snr_db = float(imp.get("impulse_snr_dB", 20.0))

    def _jakes(self, n: int) -> np.ndarray:
        angles = 2 * np.pi * np.arange(self.n_rays) / self.n_rays
        phases = 2 * np.pi * np.random.rand(self.n_rays)
        n_idx = np.arange(n)
        doppler = 2 * np.pi * self.norm_dop * np.outer(n_idx, np.cos(angles))
        h = np.sum(np.exp(1j * (doppler + phases)), axis=1) / np.sqrt(self.n_rays)
        return h / np.sqrt(np.mean(np.abs(h)**2) + 1e-12)

    def apply(self, tx: np.ndarray, snr_lin: float) -> Tuple[np.ndarray, np.ndarray]:
        n = len(tx)
        rx = tx.copy()
        h = np.ones(n, dtype=complex)

        if self.rayleigh_en or self.rician_en:
            h_sc = self._jakes(n)
            if self.rician_en:
                los = np.sqrt(self.k_lin / (self.k_lin + 1))
                h = los + np.sqrt(1.0 / (self.k_lin + 1)) * h_sc
            else:
                h = h_sc
            rx = h * rx

        sig_pwr = float(np.mean(np.abs(tx)**2)) or 1.0
        rx += _awgn_noise(n, snr_lin, sig_pwr)

        if self.imp_en:
            sigma_awgn = np.sqrt(sig_pwr / (2.0 * snr_lin))
            imp_amp = sigma_awgn * np.sqrt(10.0**(self.imp_snr_db / 10.0))
            mask = np.random.rand(n) < self.imp_prob
            if np.any(mask):
                rx[mask] += imp_amp * (np.random.randn(np.sum(mask)) + 1j * np.random.randn(np.sum(mask)))

        return rx, h