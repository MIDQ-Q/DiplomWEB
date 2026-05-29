# utils.py
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


def save_to_txt(results: List[Dict[str, Any]], config: dict) -> str:
    tmp = tempfile.mkdtemp()
    fname = Path(tmp) / "simulation_results.txt"
    mod = f"{config['modulation']['type']}-{config['modulation']['order']}"
    cod = config["coding"]["type"].upper() if config["coding"]["enabled"] else "None"

    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"Модуляция: {mod} | Код: {cod} | Gray: {config['modulation']['gray']}\n")
        imp = config['channel']['impulse_noise']
        if imp['enabled']:
            f.write(f"Канал: AWGN + Impulse(p={imp['impulse_probability']:.1e}, snr={imp['impulse_snr_dB']}дБ)\n")
        else:
            f.write("Канал: AWGN\n")
        f.write(f"{'SNR(dB)':<10} {'BER':<12} {'SER':<12} {'Theo BER':<12} {'Theo SER':<12}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(
                f"{r['snr']:<10.1f} {r['ber']:<12.2e} {r['ser']:<12.2e} {r['theo_ber']:<12.2e} {r['theo_ser']:<12.2e}\n")
    return str(fname)


def plot_ber_ser(results: List[Dict[str, Any]]) -> plt.Figure:
    snr = [r["snr"] for r in results]
    ber = [r["ber"] for r in results]
    ser = [r["ser"] for r in results]
    tb = [r["theo_ber"] for r in results]
    ts = [r["theo_ser"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr, np.clip(ber, 1e-9, 1), "o-", color="#0055ff", label="BER (эксп.)")
    ax.semilogy(snr, np.clip(ser, 1e-9, 1), "s--", color="#ff3333", label="SER (эксп.)")
    if any(v > 0 for v in tb):
        ax.semilogy(snr, np.clip(tb, 1e-9, 1), ":", color="#00aa00", label="BER (теор.)")
    if any(v > 0 for v in ts):
        ax.semilogy(snr, np.clip(ts, 1e-9, 1), "-.", color="#ffaa00", label="SER (теор.)")
    ax.set_ylim(1e-6, 1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Eb/N0 (дБ)")
    ax.set_ylabel("Вероятность ошибки")
    ax.legend()
    fig.tight_layout()
    return fig