import numpy as np
import logging
from typing import Dict, Any, Tuple
from modulation import (
    PSKModulator, QAMModulator,
    theoretical_ber_psk, theoretical_ser_psk, theoretical_ber_qam, theoretical_ser_qam,
    theoretical_ber_rayleigh_psk, theoretical_ber_rayleigh_qam
)
from channel import CompositeChannelModel
from coding import create_coder
from interleaving import get_interleaver

logger = logging.getLogger(__name__)

def _get_modulator(cfg: dict):
    t, m, gray = cfg["modulation"]["type"], cfg["modulation"]["order"], cfg["modulation"]["gray"]
    if t == "PSK": return PSKModulator(M=m, use_gray_code=gray)
    if t == "QAM": return QAMModulator(M=m, use_gray_code=gray)
    raise ValueError(f"Unknown mod: {t}")

def _theoretical_ber(cfg: dict, ebn0_db: float, is_fading: bool = False) -> float:
    t, m, gray = cfg["modulation"]["type"], cfg["modulation"]["order"], cfg["modulation"]["gray"]
    if is_fading:
        if t == "PSK": return theoretical_ber_rayleigh_psk(ebn0_db, m)
        if t == "QAM": return theoretical_ber_rayleigh_qam(ebn0_db, m)
    else:
        if t == "PSK": return theoretical_ber_psk(ebn0_db, m, gray)
        if t == "QAM": return theoretical_ber_qam(ebn0_db, m, gray)
    return 0.0

def _theoretical_ser(cfg: dict, ebn0_db: float, is_fading: bool = False) -> float:
    if is_fading: return 0.0
    return _theoretical_ber(cfg, ebn0_db, False) * np.log2(cfg["modulation"]["order"])

def _run_single_pass(cfg: dict, snr: float, use_il: bool = True) -> Tuple[int, int]:
    mod = _get_modulator(cfg)
    coder, cr = create_coder(cfg["coding"])
    ebn0_lin = 10 ** (snr / 10)
    snr_lin = ebn0_lin * cr * mod.bps
    base_bits = int(cfg.get("num_bits", 50000))
    bits = np.random.randint(0, 2, base_bits, dtype=np.uint8)
    tx_bits = coder.encode(bits) if coder else bits.copy()

    encoded_len = len(tx_bits)
    if use_il and cfg.get("interleaving", {}).get("enabled", False):
        il = get_interleaver(cfg)
        if il: tx_bits = il.interleave(tx_bits)

    tx_sym = mod.modulate(tx_bits)
    channel = CompositeChannelModel(cfg["channel"])
    rx_sym, h = channel.apply(tx_sym, snr_lin)
    rx_bits = mod.demodulate(rx_sym, h)
    rx_bits = rx_bits[:len(tx_bits)]

    if use_il and cfg.get("interleaving", {}).get("enabled", False):
        il = get_interleaver(cfg)
        if il: rx_bits = il.deinterleave(rx_bits, original_len=encoded_len)

    dec_bits, _ = coder.decode(rx_bits) if coder else (rx_bits, {})
    dec_bits = dec_bits[:len(bits)]
    errs = int(np.sum(bits != dec_bits))
    return errs, len(bits)

def simulate_transmission(cfg: dict, snr: float) -> Dict[str, Any]:
    is_fading = cfg["channel"].get("rayleigh", {}).get("enabled") or cfg["channel"].get("rician", {}).get("enabled")
    num_sims = max(1, int(cfg.get("num_simulations", 1)))

    total_errs, total_bits = 0, 0
    for _ in range(num_sims):
        e, b = _run_single_pass(cfg, snr, use_il=True)
        total_errs += e
        total_bits += b

    ber = total_errs / total_bits if total_bits > 0 else 0.0
    ser = 1.0 - (1.0 - ber) ** int(np.log2(cfg["modulation"]["order"])) if ber < 1.0 else 1.0

    res = {
        "snr": float(snr), "ber": float(ber), "ser": float(ser), "bit_errors": total_errs,
        "theoretical_ber": float(_theoretical_ber(cfg, snr, is_fading)),
        "theoretical_ser": float(_theoretical_ser(cfg, snr, is_fading))
    }

    if cfg.get("show_il_gain", False):
        cfg_no_il = cfg.copy()
        cfg_no_il["interleaving"] = {"enabled": False}
        cfg_no_il["num_simulations"] = 1
        e_no, b_no = _run_single_pass(cfg_no_il, snr, use_il=False)
        res["ber_no_il"] = float(e_no / b_no if b_no > 0 else 0.0)

    return res