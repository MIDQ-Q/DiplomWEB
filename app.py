import streamlit as st
import numpy as np
import sys, os, time, json, matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from simulation import simulate_transmission
    from coding import rs_available
except ImportError as e:
    st.error(f"Ошибка импорта ядра: {e}")
    st.stop()

st.set_page_config(page_title="Симулятор цифровой связи", layout="wide", initial_sidebar_state="expanded")

if "comparison_runs" not in st.session_state: st.session_state.comparison_runs = []
if "last_results" not in st.session_state: st.session_state.last_results = None
if "last_config" not in st.session_state: st.session_state.last_config = None

def build_config(params):
    snr_start, snr_stop, snr_step = params["snr_start"], params["snr_stop"], params["snr_step"]
    if snr_start >= snr_stop: raise ValueError("Начальное значение должно быть меньше конечного")
    ebn0_range = []
    val = snr_start
    while val <= snr_stop + 1e-9:
        ebn0_range.append(round(val, 4))
        val += snr_step

    ch_cfg = {"awgn": {"enabled": True}, "rayleigh": {"enabled": False}, "rician": {"enabled": False}, "impulse_noise": {"enabled": False}}
    if params["ch_type"] == "Rayleigh":
        ch_cfg["rayleigh"] = {"enabled": True, "n_rays": int(params["n_rays"]), "normalized_doppler": float(params["norm_dop"])}
    elif params["ch_type"] == "Rician":
        ch_cfg["rician"] = {"enabled": True, "k_factor_dB": float(params["k_db"]), "n_rays": int(params["n_rays"]), "normalized_doppler": float(params["norm_dop"])}

    if params["imp_enabled"]:
        ch_cfg["impulse_noise"] = {"enabled": True, "impulse_probability": float(params["imp_prob"]), "impulse_snr_dB": float(params["imp_snr"])}

    il_cfg = {"enabled": params["il_enabled"]}
    if params["il_enabled"]:
        if params["il_auto"] and params["coding_type"] in ("reed-solomon", "rs"):
            il_cfg["depth"] = None
            il_cfg["expected_burst_bits"] = int(params["il_burst"])
        else:
            il_cfg["depth"] = int(params["il_depth"])

    return {
        "modulation": {"type": params["mod_type"], "order": int(params["mod_order"]), "gray": bool(params["gray"])},
        "coding": {"enabled": bool(params["coding_on"]), "type": params["coding_type"] if params["coding_on"] else "none", **params["cod_params"]},
        "channel": ch_cfg,
        "interleaving": il_cfg,
        "ebn0_dB_range": ebn0_range,
        "num_bits": int(params["num_bits"]),
        "num_simulations": int(params["num_sims"]),
        "show_il_gain": bool(params["show_il_gain"])
    }

def plot_results(results, config, show_comparison=False, width_scale=1.0):
    snr = [r["snr"] for r in results]
    ber = [r["ber"] for r in results]
    ser = [r["ser"] for r in results]
    tb = [r.get("theoretical_ber", 0) for r in results]
    ts = [r.get("theoretical_ser", 0) for r in results]

    fig, ax = plt.subplots(figsize=(9 * width_scale, 5))
    ax.semilogy(snr, np.clip(ber, 1e-9, 1), "o-", color="#1D63ED", label="BER (эксперимент)")
    ax.semilogy(snr, np.clip(ser, 1e-9, 1), "s--", color="#C42B1C", label="SER (эксперимент)")

    if show_comparison and "ber_no_il" in results[0]:
        ber_no = [r["ber_no_il"] for r in results]
        ax.semilogy(snr, np.clip(ber_no, 1e-9, 1), "v-", color="#FF9900", alpha=0.7, label="BER (без перемежения)")

    if any(v > 0 for v in tb): ax.semilogy(snr, np.clip(tb, 1e-9, 1), ":", color="#107C10", label="Теоретический BER")
    if any(v > 0 for v in ts): ax.semilogy(snr, np.clip(ts, 1e-9, 1), "-.", color="#FFB900", label="Теоретический SER")

    ax.set_ylim(1e-6, 1)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Отношение Eb/N0 (дБ)")
    ax.set_ylabel("Вероятность ошибки")

    ch_names = []
    if config["channel"].get("rayleigh", {}).get("enabled"): ch_names.append("Релеевские замирания")
    if config["channel"].get("rician", {}).get("enabled"): ch_names.append("Райсовские замирания")
    if config["channel"].get("impulse_noise", {}).get("enabled"): ch_names.append("Импульсные помехи")
    title_ch = "АБГШ" if not ch_names else f"АБГШ + {', '.join(ch_names)}"
    ax.set_title(f"{config['modulation']['type']}-{config['modulation']['order']} | Канал: {title_ch} | Усреднений: {config['num_simulations']}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

def plot_comparison(runs, width_scale=1.0):
    fig, ax = plt.subplots(figsize=(9 * width_scale, 5))
    colors = ["#1D63ED", "#C42B1C", "#107C10", "#FFB900", "#9900FF", "#00CC99"]
    for i, run in enumerate(runs):
        res, cfg, label = run["results"], run["config"], run["label"]
        snr = [r["snr"] for r in res]
        ber = [r["ber"] for r in res]
        c = colors[i % len(colors)]
        ax.semilogy(snr, np.clip(ber, 1e-9, 1), "o-", color=c, label=f"{label} (BER)")
    ax.set_ylim(1e-6, 1)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Отношение Eb/N0 (дБ)")
    ax.set_ylabel("Вероятность ошибки")
    ax.set_title("Сравнение независимых прогонов")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

st.title("Симулятор цифровой связи")
with st.sidebar:
    st.header("Настройки модуляции")
    mod_type = st.selectbox("Тип модуляции", ["PSK", "QAM"])
    mod_orders = {"PSK": [2, 4, 8, 16], "QAM": [4, 16, 64]}
    mod_order = st.selectbox("Порядок модуляции M", options=mod_orders[mod_type], index=1 if mod_type=="PSK" else 1)
    gray_code = st.checkbox("Использовать код Грея", value=True)

    st.header("Параметры канала")
    ch_options = {"АБГШ (без замираний)": "AWGN", "Релеевские замирания": "Rayleigh", "Райсовские замирания": "Rician"}
    ch_display = st.radio("Тип канала", list(ch_options.keys()), index=0)
    ch_type = ch_options[ch_display]
    n_rays, norm_dop, k_db = 16, 0.01, 3.0
    if ch_type != "AWGN":
        n_rays = st.number_input("Число лучей рассеяния", 4, 64, 16)
        norm_dop = st.number_input("Нормированная доплеровская частота", 0.001, 0.1, 0.01, 0.001)
        if ch_type == "Rician":
            k_db = st.number_input("Фактор К (дБ)", 0.0, 20.0, 3.0)
    imp_enabled = st.checkbox("Добавить импульсные помехи", value=False)
    imp_prob, imp_snr = 0.001, 20.0
    if imp_enabled:
        c1, c2 = st.columns(2)
        imp_prob = c1.number_input("Вероятность появления помехи (p)", min_value=1e-6, max_value=0.1, value=0.001, format="%.4f")
        imp_snr = c2.number_input("Отношение сигнал/помеха импульса (дБ)", min_value=0.0, max_value=40.0, value=20.0)

    st.header("Перемежение")
    il_enabled = st.checkbox("Включить блочное перемежение", value=False)
    il_depth = 8
    il_burst = 32
    il_auto = False
    if il_enabled:
        il_auto = st.checkbox("Автоматический расчёт глубины для кодов Рида-Соломона", value=True)
        if not il_auto:
            il_depth = st.number_input("Глубина перемежения (D)", min_value=2, max_value=256, value=8)
        else:
            il_burst = st.slider("Ожидаемая длина пакетной ошибки (бит)", 8, 256, 32)

    st.header("Канальное кодирование")
    coding_on = st.checkbox("Включить кодирование", value=False)
    coding_type = "none"
    cod_params = {}
    if coding_on:
        coding_type = st.radio("Тип кода", ["Хэмминг", "Рида-Соломона"])
        if coding_type == "Хэмминг":
            preset = st.selectbox("Пресет кода Хэмминга (n,k)", ["(7,4)", "(15,11)", "(31,26)", "(63,57)", "(127,120)"])
            n, k = {"(7,4)": (7,4), "(15,11)": (15,11), "(31,26)": (31,26), "(63,57)": (63,57), "(127,120)": (127,120)}[preset]
            cod_params = {"type": "hamming", "n": n, "k": k}
        else:
            if not rs_available(): st.warning("Библиотека reedsolo не установлена"); coding_on = False
            else:
                cod_params = {"type": "reed-solomon", "rs_nsym": int(st.number_input("Количество контрольных символов (nsym)", value=10)), "rs_nsize": int(st.number_input("Размер блока кода (nsize, макс. 255)", min_value=10, max_value=255, value=255))}

    st.header("Диапазон Eb/N0 и параметры симуляции")
    c1, c2, c3 = st.columns(3)
    snr_start = c1.number_input("Начало диапазона", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    snr_stop = c2.number_input("Конец диапазона", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
    snr_step = c3.number_input("Шаг сетки", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    num_bits = st.slider("Количество бит в блоке", min_value=10_000, max_value=1_000_000, value=50_000, step=10_000)
    if num_bits > 500_000: st.warning("Обработка более 500 тыс. бит может занять значительное время.")

    num_sims = st.selectbox("Количество независимых прогонов (усреднение)", [1, 2, 3, 4, 5], index=0)

    show_il_gain = st.checkbox("Отобразить сравнение BER с перемежением и без", value=False) if il_enabled else False

    st.divider()
    add_to_comp = st.checkbox("Сохранить в список для сравнения")
    run_label = st.text_input("Метка для легенды графика", placeholder="Например: QPSK + RS(10) + IL")

    if st.button("Очистить список сохранённых прогонов", type="secondary"):
        if st.session_state.comparison_runs:
            st.session_state.comparison_runs = []
            st.rerun()

    if st.session_state.comparison_runs: st.write(f"В списке сравнения: {len(st.session_state.comparison_runs)}")
    run_btn = st.button("Начать расчёт", type="primary", use_container_width=True)

if run_btn:
    try:
        config = build_config({
            "mod_type": mod_type, "mod_order": mod_order, "gray": gray_code,
            "ch_type": ch_type, "n_rays": n_rays, "norm_dop": norm_dop, "k_db": k_db,
            "imp_enabled": imp_enabled, "imp_prob": imp_prob, "imp_snr": imp_snr,
            "il_enabled": il_enabled, "il_auto": il_auto, "il_depth": il_depth, "il_burst": il_burst,
            "coding_on": coding_on, "coding_type": coding_type.lower().replace("хэмминг", "hamming").replace("рида-соломона", "reed-solomon"),
            "cod_params": cod_params,
            "snr_start": snr_start, "snr_stop": snr_stop, "snr_step": snr_step,
            "num_bits": num_bits, "num_sims": num_sims, "show_il_gain": show_il_gain
        })
    except ValueError as e:
        st.error(f"Ошибка конфигурации: {e}")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    t_start = time.time()

    for i, snr in enumerate(config["ebn0_dB_range"]):
        status_text.text(f"Расчёт точки Eb/N0 = {snr:.1f} дБ ({i + 1}/{len(config['ebn0_dB_range'])})")
        try:
            results.append(simulate_transmission(config, snr))
        except Exception as e:
            status_text.error(f"Ошибка вычисления: {e}")
            break
        progress_bar.progress((i + 1) / len(config["ebn0_dB_range"]))

    if results:
        st.session_state.last_results = results
        st.session_state.last_config = config
        exec_time = time.time() - t_start
        status_text.success(f"Расчёт завершён за {exec_time:.1f} сек")
        progress_bar.empty()
        if add_to_comp:
            label = run_label.strip() or f"Прогон {len(st.session_state.comparison_runs) + 1}"
            st.session_state.comparison_runs.append({"config": config, "results": results, "label": label})
            st.success(f"Добавлено в список сравнения: `{label}`")
            st.rerun()

if st.session_state.last_results and st.session_state.last_config:
    res, cfg = st.session_state.last_results, st.session_state.last_config
    st.subheader("Результаты последнего расчёта")
    st.dataframe([{"Eb/N0 (дБ)": f"{r['snr']:.1f}", "BER": f"{r['ber']:.2e}", "SER": f"{r['ser']:.2e}", "Число ошибок": r.get("bit_errors", 0)} for r in res], use_container_width=True, hide_index=True)
    st.subheader("График зависимости вероятности ошибки")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.pyplot(plot_results(res, cfg, show_comparison=cfg.get("show_il_gain", False), width_scale=1.0), use_container_width=True)

    st.subheader("Сохранение результатов")
    cfg_header = "# КОНФИГУРАЦИЯ РАСЧЁТА\n" + json.dumps(cfg, indent=2, ensure_ascii=False) + "\n# ДАННЫЕ (Eb/N0 BER SER)\n"
    txt_data = cfg_header + "\n".join([f"{r['snr']:.1f} {r['ber']:.2e} {r['ser']:.2e}" for r in res])
    st.download_button("Экспортировать в текстовый файл", data=txt_data, file_name=f"sim_{int(time.time())}.txt", mime="text/plain", use_container_width=True)

if st.session_state.comparison_runs:
    st.divider()
    st.subheader("Совмещённый график сравнения")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.pyplot(plot_comparison(st.session_state.comparison_runs, width_scale=1.0), use_container_width=True)
    if st.button("Удалить последний из списка"): st.session_state.comparison_runs.pop(); st.rerun()

if not st.session_state.last_results and not st.session_state.comparison_runs:
    st.info("Настройте параметры в боковой панели и нажмите **Начать расчёт**")