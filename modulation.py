import numpy as np
import scipy.special as sp
from math import pi, log2
from functools import lru_cache

@lru_cache(maxsize=512)
def Q_function(x: float) -> float:
    if x < 0: return 1.0 - Q_function(-x)
    return float(0.5 * sp.erfc(x / np.sqrt(2)))

class PSKModulator:
    def __init__(self, M: int = 4, use_gray_code: bool = True) -> None:
        if M not in (2, 4, 8, 16): raise ValueError(f"PSK: M ∈ {{2,4,8,16}}, получено {M}")
        self.M, self.bps = M, int(log2(M))
        self.use_gray_code = use_gray_code
        self._build_luts()

    def _build_luts(self) -> None:
        angles = np.linspace(0, 2*pi, self.M, endpoint=False)
        if self.M == 4: angles += pi/4
        self.constellation = np.exp(1j * angles)
        self.bit_to_idx, self.idx_to_bits = {}, {}
        for i in range(self.M):
            code = i ^ (i >> 1) if self.use_gray_code else i
            bits = tuple((code >> (self.bps - 1 - j)) & 1 for j in range(self.bps))
            self.bit_to_idx[bits], self.idx_to_bits[i] = i, bits
        self._bits_lut = np.array([self.idx_to_bits[i] for i in range(self.M)], dtype=np.uint8)
        self._mod_lut = np.argsort(np.arange(self.M) ^ (np.arange(self.M) >> 1)).astype(np.int32) if self.use_gray_code else np.arange(self.M, dtype=np.int32)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(b)) % self.bps
        if pad: b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
        groups = b.reshape(-1, self.bps)
        powers = 1 << np.arange(self.bps - 1, -1, -1, dtype=np.int32)
        nat = groups.astype(np.int32) @ powers
        return self.constellation[self._mod_lut[nat]]

    def demodulate(self, symbols: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        s = np.asarray(symbols, dtype=complex).ravel()
        if h is not None:
            h = np.asarray(h, dtype=complex).ravel()
            s = s * np.exp(-1j * np.angle(h))
        dist = np.abs(s[:, None] - self.constellation[None, :])**2
        return self._bits_lut[np.argmin(dist, axis=1)].ravel()

class QAMModulator:
    def __init__(self, M: int = 16, use_gray_code: bool = True) -> None:
        if M not in (4, 16, 64, 256): raise ValueError(f"QAM: M ∈ {{4,16,64,256}}, получено {M}")
        self.M, self.bps = M, int(log2(M))
        self.use_gray_code = use_gray_code
        self._build_luts()

    def _build_luts(self) -> None:
        side = int(np.sqrt(self.M))
        amps = np.arange(-(side-1), side, 2, dtype=float)
        g = np.array([i ^ (i>>1) for i in range(side)], dtype=np.int32)
        I, Q = np.meshgrid(amps, amps)
        pts = (I + 1j*Q).ravel()
        pts /= np.sqrt(np.mean(np.abs(pts)**2))
        self.constellation = pts
        half = self.bps // 2
        self.bit_to_idx, self.idx_to_bits = {}, {}
        for i in range(self.M):
            qi, ii = i//side, i%side
            gi, gq = (g[ii], g[qi]) if self.use_gray_code else (ii, qi)
            bits = tuple((gi>>(half-1-j))&1 for j in range(half)) + tuple((gq>>(half-1-j))&1 for j in range(half))
            self.bit_to_idx[bits], self.idx_to_bits[i] = i, bits
        self._bits_lut = np.array([self.idx_to_bits[i] for i in range(self.M)], dtype=np.uint8)
        bps = self.bps
        self._mod_lut = np.empty(self.M, dtype=np.int32)
        for nat in range(self.M):
            self._mod_lut[nat] = self.bit_to_idx[tuple((nat>>(bps-1-j))&1 for j in range(bps))]

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(b)) % self.bps
        if pad: b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
        groups = b.reshape(-1, self.bps)
        powers = 1 << np.arange(self.bps-1, -1, -1, dtype=np.int32)
        nat = groups.astype(np.int32) @ powers
        return self.constellation[self._mod_lut[nat]]

    def demodulate(self, symbols: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        s = np.asarray(symbols, dtype=complex).ravel()
        if h is not None:
            h = np.asarray(h, dtype=complex).ravel()
            with np.errstate(divide='ignore', invalid='ignore'):
                s = s / h
            bad = ~np.isfinite(s)
            if np.any(bad): s[bad] = symbols[bad][bad]
        dist = np.abs(s[:, None] - self.constellation[None, :])**2
        return self._bits_lut[np.argmin(dist, axis=1)].ravel()

def theoretical_ber_psk(ebn0_dB: float, M: int, gray: bool = True) -> float:
    ebn0 = 10**(ebn0_dB/10); k = log2(M)
    if M in (2,4): return Q_function(float(np.sqrt(2*ebn0)))
    return theoretical_ser_psk(ebn0_dB, M, gray) / k

def theoretical_ser_psk(ebn0_dB: float, M: int, gray: bool = True) -> float:
    ebn0 = 10**(ebn0_dB/10); k = log2(M)
    if M==2: return Q_function(float(np.sqrt(2*ebn0)))
    if M==4: q=Q_function(float(np.sqrt(2*ebn0))); return 2*q - q**2
    return 2*Q_function(float(np.sqrt(2*k*ebn0)*np.sin(pi/M)))

def theoretical_ber_qam(ebn0_dB: float, M: int, gray: bool = True) -> float:
    ebn0 = 10**(ebn0_dB/10); k = log2(M)
    if M==4: return Q_function(float(np.sqrt(2*ebn0)))
    return float((4/k)*(1-1/np.sqrt(M))*Q_function(float(np.sqrt(3*k*ebn0/(M-1)))))

def theoretical_ser_qam(ebn0_dB: float, M: int, gray: bool = True) -> float:
    ebn0 = 10**(ebn0_dB/10); k = log2(M)
    if M==4: q=Q_function(float(np.sqrt(2*ebn0))); return 2*q-q**2
    q = Q_function(float(np.sqrt(3*k*ebn0/(M-1))))
    c = 1-1/np.sqrt(M)
    return float(4*c*q - 4*c**2*q**2)

def theoretical_ber_rayleigh_psk(ebn0_dB: float, M: int) -> float:
    ebn0 = 10**(ebn0_dB/10); k = log2(M)
    if M == 2: return 0.5 * (1 - np.sqrt(ebn0/(1+ebn0)))
    if M == 4: return 0.5 * (1 - np.sqrt(ebn0/(1+ebn0)))
    gamma_s = ebn0 * k
    alpha = gamma_s * np.sin(pi/M)**2 / (1 + gamma_s * np.sin(pi/M)**2)
    return float(np.clip(2*(M-1)/M * 0.5 * (1 - np.sqrt(alpha/(1+alpha))) / k, 0, 0.5))

def theoretical_ber_rayleigh_qam(ebn0_dB: float, M: int) -> float:
    ebn0 = 10**(ebn0_dB/10); k = log2(M)
    if M == 4: return theoretical_ber_rayleigh_psk(ebn0_dB, 4)
    beta = 3*k*ebn0 / (2*(M-1))
    p_e = 0.5 * (1 - np.sqrt(beta/(1+beta)))
    return float(np.clip((4/k)*(1-1/np.sqrt(M))*p_e, 0, 0.5))