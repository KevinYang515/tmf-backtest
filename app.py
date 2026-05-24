"""
TMF 微型台指期貨 — 回測結果儀表板
策略 A：波動突破
策略 B：固定時間進多 + 高掛 Limit
策略 C：前日強勢大型股（日頻上限估算）
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="TMF 回測儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* global dark tone */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stHeader"]           { background: transparent; }

/* metric cards */
.kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 4px 0;
    text-align: center;
}
.kpi-label { color: #8b949e; font-size: 13px; margin-bottom: 4px; }
.kpi-value { color: #e6edf3; font-size: 28px; font-weight: 700; }
.kpi-sub   { color: #8b949e; font-size: 12px; margin-top: 2px; }

/* green/red/yellow */
.pos  { color: #3fb950 !important; }
.neg  { color: #f85149 !important; }
.neu  { color: #d29922 !important; }
.dim  { color: #8b949e !important; }

/* section header */
.sec-header {
    border-left: 4px solid #388bfd;
    padding-left: 10px;
    color: #e6edf3;
    font-size: 16px;
    font-weight: 600;
    margin: 18px 0 10px;
}

/* dataframe tweaks */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(ttl=300)
def load_breakout():
    p = DATA_DIR / "results_breakout.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    return df


@st.cache_data(ttl=300)
def load_scalp():
    p = DATA_DIR / "results_scalp.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    return df


@st.cache_data(ttl=300)
def load_stock_scalp():
    p = DATA_DIR / "results_stock_scalp.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    return df


def kpi_card(label: str, value: str, sub: str = "", cls: str = ""):
    color_cls = f' {cls}' if cls else ''
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value{color_cls}">{value}</div>
        {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
    </div>"""


def color_pnl(val):
    try:
        v = float(val)
        if v > 0:   return "color: #3fb950; font-weight: 600"
        elif v < 0: return "color: #f85149; font-weight: 600"
    except Exception:
        pass
    return ""


def color_sharpe(val):
    try:
        v = float(val)
        if v >= 5:   return "color: #3fb950; font-weight: 700"
        elif v >= 2: return "color: #d29922; font-weight: 600"
        elif v < 0:  return "color: #f85149"
    except Exception:
        pass
    return ""


def color_wr(val):
    try:
        v = float(str(val).replace("%", ""))
        if v >= 60:   return "color: #3fb950"
        elif v >= 50: return "color: #d29922"
        else:         return "color: #f85149"
    except Exception:
        pass
    return ""


def fmt_pnl(x):
    try:
        v = int(float(x))
        return f"{v:+,}"
    except Exception:
        return x


def monthly_summary(tdf: pd.DataFrame, pnl_col="pnl_twd") -> pd.DataFrame:
    """共用月份穩定性彙總（策略 A/B 用）"""
    rows = []
    for month, mdf in tdf.groupby("month"):
        wins = (mdf[pnl_col] > 0).sum() if pnl_col in mdf.columns else 0
        rows.append({
            "月份": month,
            "交易日": len(mdf),
            "勝率": f"{wins/len(mdf)*100:.1f}%",
            "損益(元)": int(mdf[pnl_col].sum()),
            "平均損益": round(mdf[pnl_col].mean(), 0),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# Main layout
# ════════════════════════════════════════════════════════════════════

st.markdown("## 📊 TMF 微型台指期貨 — 回測儀表板")
st.caption("資料期間：2026-02-23 ~ 2026-05-23（74 個交易日）｜手續費：NT$40 round-trip（4 點）")

tab_a, tab_b, tab_c = st.tabs(["策略 A：波動突破", "策略 B：固定時間進多", "策略 C：前日強勢大型股"])


# ════════════════════════════════════════════════════════════════════
# TAB A：波動突破
# ════════════════════════════════════════════════════════════════════
with tab_a:
    st.markdown("""
    <div class="sec-header">策略邏輯</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - **觸發時間**：08:46 / 09:00 / 13:30 / 13:45（K 棒開盤）
    - **進場條件**：突破 `entry ± offset` 點
    - **停利**：觸發後 `target` 點
    - **停損**：觸發後 `-stop` 點（time limit 到收盤平）
    - **參數空間**：3,920 組組合
    """)

    rdf = load_breakout()
    if rdf.empty:
        st.info("回測資料尚未載入")
    else:
        # ── 全局最佳 KPI ──
        best = rdf.loc[rdf["sharpe"].idxmax()]
        total_pnl_best = int(best["total_pnl"])
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            ("最佳 Sharpe", f"{best['sharpe']:.2f}", f"{best['trigger']}  offset={int(best['offset'])} tgt={int(best['target'])} stp={int(best['stop'])} tlim={int(best['time_limit'])}", "pos"),
            ("最佳總損益", f"{total_pnl_best:+,} 元", "最高 Sharpe 組合", "pos" if total_pnl_best > 0 else "neg"),
            ("勝率", f"{best['win_rate']:.1f}%", "最高 Sharpe 組合", ""),
            ("交易日", f"{int(best['trades'])}", "74 個交易日中", ""),
            ("策略組數", f"{len(rdf):,}", "正 Sharpe", f"{(rdf['sharpe']>0).sum()}"),
        ]
        for col, (label, val, sub, cls) in zip([c1,c2,c3,c4,c5], kpis):
            col.markdown(kpi_card(label, val, sub, cls), unsafe_allow_html=True)

        st.markdown('<div class="sec-header">篩選器</div>', unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        triggers = ["全部"] + sorted(rdf["trigger"].unique().tolist())
        sel_trigger = col_f1.selectbox("觸發時間", triggers, key="a_trigger")
        min_sharpe = col_f2.slider("最低 Sharpe", -5.0, 15.0, 0.0, 0.5, key="a_sharpe")
        min_wr = col_f3.slider("最低勝率 (%)", 0, 100, 0, 5, key="a_wr")

        fdf = rdf.copy()
        if sel_trigger != "全部":
            fdf = fdf[fdf["trigger"] == sel_trigger]
        fdf = fdf[(fdf["sharpe"] >= min_sharpe) & (fdf["win_rate"] >= min_wr)]

        st.markdown(f'<div class="sec-header">Top 30（Sharpe，共 {len(fdf):,} 組符合）</div>', unsafe_allow_html=True)

        COLS_A = ["trigger", "offset", "target", "stop", "time_limit",
                  "trades", "win_rate", "total_pnl", "avg_pnl", "sharpe"]
        display_a = fdf.sort_values("sharpe", ascending=False).head(30)[
            [c for c in COLS_A if c in fdf.columns]
        ].copy()
        display_a["total_pnl"] = display_a["total_pnl"].apply(fmt_pnl)
        display_a["avg_pnl"]   = display_a["avg_pnl"].apply(fmt_pnl)
        display_a["win_rate"]  = display_a["win_rate"].apply(lambda x: f"{x:.1f}%")
        display_a.columns = [c.replace("_", " ").title() for c in display_a.columns]

        styled_a = display_a.style \
            .map(color_sharpe, subset=["Sharpe"]) \
            .map(color_wr, subset=["Win Rate"]) \
            .map(color_pnl, subset=["Total Pnl", "Avg Pnl"])
        st.dataframe(styled_a, use_container_width=True, hide_index=True, height=500)

        # ── 各觸發時間最佳 ──
        st.markdown('<div class="sec-header">各觸發時間最佳組合</div>', unsafe_allow_html=True)
        best_per = rdf.loc[rdf.groupby("trigger")["sharpe"].idxmax()][
            [c for c in COLS_A if c in rdf.columns]
        ].copy()
        best_per["total_pnl"] = best_per["total_pnl"].apply(fmt_pnl)
        best_per["avg_pnl"]   = best_per["avg_pnl"].apply(fmt_pnl)
        best_per["win_rate"]  = best_per["win_rate"].apply(lambda x: f"{x:.1f}%")
        best_per.columns = [c.replace("_", " ").title() for c in best_per.columns]
        styled_bp = best_per.style \
            .map(color_sharpe, subset=["Sharpe"]) \
            .map(color_wr, subset=["Win Rate"]) \
            .map(color_pnl, subset=["Total Pnl", "Avg Pnl"])
        st.dataframe(styled_bp, use_container_width=True, hide_index=True)

        # ── Bar chart ──
        st.markdown('<div class="sec-header">各觸發時間最佳總損益</div>', unsafe_allow_html=True)
        chart_a = rdf.loc[rdf.groupby("trigger")["sharpe"].idxmax()][["trigger", "total_pnl"]].set_index("trigger")
        st.bar_chart(chart_a, height=280)

        st.caption("⚠️ 以上為參數掃描上限估計，請注意過擬合風險")


# ════════════════════════════════════════════════════════════════════
# TAB B：固定時間進多
# ════════════════════════════════════════════════════════════════════
with tab_b:
    st.markdown('<div class="sec-header">策略邏輯</div>', unsafe_allow_html=True)
    st.markdown("""
    - **進場**：固定時間無條件進多 1 口
    - **08:46 觸發**：方向濾網 = 夜盤（前日 15:00 ~ 當日 05:00）偏多才進
    - **09:00 觸發**：方向濾網 = 08:46–08:59 偏多才進
    - **停利**：掛 Limit sell 在 entry + target
    - **停損**：選配（無停損 = 純時間停損）
    - **時間停損**：time_limit 根 K 棒後市價平倉
    """)

    rdf_b = load_scalp()
    if rdf_b.empty:
        st.info("回測資料尚未載入")
    else:
        # ── 全局最佳 KPI ──
        best_b = rdf_b.loc[rdf_b["sharpe"].idxmax()]
        total_pnl_b = int(best_b["total_pnl"])
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis_b = [
            ("最佳 Sharpe",   f"{best_b['sharpe']:.2f}",  f"{best_b['trigger']}  tgt={int(best_b['target'])} stp={int(best_b['stop'])} tlim={int(best_b['time_limit'])}", "pos"),
            ("最佳總損益",    f"{total_pnl_b:+,} 元",     "最高 Sharpe 組合", "pos" if total_pnl_b > 0 else "neg"),
            ("TP 率",         f"{best_b['tp_rate']:.1f}%", "最高 Sharpe 組合", "pos"),
            ("平均 TP 損益",  f"{int(best_b['avg_pnl']):+,} 元", "", "pos" if best_b['avg_pnl'] > 0 else "neg"),
            ("正 Sharpe 組數",f"{(rdf_b['sharpe']>0).sum()}", f"共 {len(rdf_b)} 組", ""),
        ]
        for col, (label, val, sub, cls) in zip([c1,c2,c3,c4,c5], kpis_b):
            col.markdown(kpi_card(label, val, sub, cls), unsafe_allow_html=True)

        st.markdown('<div class="sec-header">篩選器</div>', unsafe_allow_html=True)
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        triggers_b = ["全部"] + sorted(rdf_b["trigger"].unique().tolist())
        sel_trig_b = col_f1.selectbox("觸發時間", triggers_b, key="b_trigger")
        sel_stop   = col_f2.selectbox("停損設定", ["全部", "無停損(0)", "有停損"], key="b_stop")
        min_shr_b  = col_f3.slider("最低 Sharpe", -10.0, 15.0, 0.0, 0.5, key="b_sharpe")
        min_tp_b   = col_f4.slider("最低 TP 率 (%)", 0, 100, 0, 5, key="b_tp")

        fdf_b = rdf_b.copy()
        if sel_trig_b != "全部":
            fdf_b = fdf_b[fdf_b["trigger"] == sel_trig_b]
        if sel_stop == "無停損(0)":
            fdf_b = fdf_b[fdf_b["stop"] == 0]
        elif sel_stop == "有停損":
            fdf_b = fdf_b[fdf_b["stop"] > 0]
        fdf_b = fdf_b[(fdf_b["sharpe"] >= min_shr_b) & (fdf_b["tp_rate"] >= min_tp_b)]

        COLS_B = ["trigger", "target", "stop", "time_limit", "trades",
                  "tp_rate", "sl_rate", "timeout_rate", "win_rate",
                  "total_pnl", "avg_pnl", "loss_avg", "sharpe"]

        st.markdown(f'<div class="sec-header">Top 30（Sharpe，共 {len(fdf_b):,} 組符合）</div>', unsafe_allow_html=True)
        display_b = fdf_b.sort_values("sharpe", ascending=False).head(30)[
            [c for c in COLS_B if c in fdf_b.columns]
        ].copy()
        for col_ in ["total_pnl", "avg_pnl"]:
            if col_ in display_b.columns:
                display_b[col_] = display_b[col_].apply(fmt_pnl)
        for col_ in ["tp_rate", "sl_rate", "timeout_rate", "win_rate"]:
            if col_ in display_b.columns:
                display_b[col_] = display_b[col_].apply(lambda x: f"{x:.1f}%")
        display_b.columns = [c.replace("_", " ").title() for c in display_b.columns]

        style_cols_b = {}
        if "Sharpe"      in display_b.columns: style_cols_b["Sharpe"]      = color_sharpe
        if "Win Rate"    in display_b.columns: style_cols_b["Win Rate"]    = color_wr
        if "Total Pnl"   in display_b.columns: style_cols_b["Total Pnl"]   = color_pnl
        if "Avg Pnl"     in display_b.columns: style_cols_b["Avg Pnl"]     = color_pnl

        styled_b = display_b.style
        for col_, fn in style_cols_b.items():
            styled_b = styled_b.map(fn, subset=[col_])
        st.dataframe(styled_b, use_container_width=True, hide_index=True, height=500)

        # ── 分區對比 ──
        st.markdown('<div class="sec-header">策略細分對比</div>', unsafe_allow_html=True)
        col_900, col_846 = st.columns(2)

        with col_900:
            st.markdown("**09:00 × 無停損 × 各時間限制**")
            sub_900 = (rdf_b[(rdf_b["trigger"] == "現貨開盤") & (rdf_b["stop"] == 0)]
                       .sort_values("time_limit")[[c for c in COLS_B if c in rdf_b.columns]])
            sub_900 = sub_900.copy()
            for col_ in ["total_pnl", "avg_pnl"]:
                if col_ in sub_900.columns:
                    sub_900[col_] = sub_900[col_].apply(fmt_pnl)
            for col_ in ["tp_rate", "sl_rate", "timeout_rate", "win_rate"]:
                if col_ in sub_900.columns:
                    sub_900[col_] = sub_900[col_].apply(lambda x: f"{x:.1f}%")
            sub_900.columns = [c.replace("_", " ").title() for c in sub_900.columns]
            st.dataframe(sub_900, use_container_width=True, hide_index=True)

        with col_846:
            st.markdown("**08:46 × time_limit=5 × 各停損設定**")
            sub_846 = (rdf_b[(rdf_b["trigger"] == "期貨開盤") & (rdf_b["time_limit"] == 5)]
                       .sort_values("stop")[[c for c in COLS_B if c in rdf_b.columns]])
            sub_846 = sub_846.copy()
            for col_ in ["total_pnl", "avg_pnl"]:
                if col_ in sub_846.columns:
                    sub_846[col_] = sub_846[col_].apply(fmt_pnl)
            for col_ in ["tp_rate", "sl_rate", "timeout_rate", "win_rate"]:
                if col_ in sub_846.columns:
                    sub_846[col_] = sub_846[col_].apply(lambda x: f"{x:.1f}%")
            sub_846.columns = [c.replace("_", " ").title() for c in sub_846.columns]
            st.dataframe(sub_846, use_container_width=True, hide_index=True)

        # ── 月份穩定性 ──
        with st.expander("月份穩定性（展開）"):
            best_params = rdf_b.sort_values("sharpe", ascending=False).iloc[0]
            trig_best   = best_params["trigger"]
            tgt_best    = int(best_params["target"])
            stp_best    = int(best_params["stop"])
            tlim_best   = int(best_params["time_limit"])

            # 重新跑月份彙總用 trades 詳細資料（此處 CSV 只有彙總，直接載入 stock_scalp 的 date/pnl）
            # 若無 date 欄，只顯示說明
            if "date" not in rdf_b.columns:
                st.caption(f"最佳組合：{trig_best}  target={tgt_best}  stop={stp_best}  time_limit={tlim_best}")
                st.info("月份穩定性需要 trades 詳細 CSV（目前只有彙總）。")
            else:
                st.caption(f"最佳組合：{trig_best}  target={tgt_best}  stop={stp_best}  time_limit={tlim_best}")

        st.caption("⚠️ 夜盤方向濾網為合法前置資訊（前日 15:00～當日 05:00）")


# ════════════════════════════════════════════════════════════════════
# TAB C：前日強勢大型股
# ════════════════════════════════════════════════════════════════════
with tab_c:
    st.markdown('<div class="sec-header">策略邏輯</div>', unsafe_allow_html=True)
    st.markdown("""
    - **選股**：前一日漲幅前 30 名（市值 > 500 億，價格 100-200 或 1000-2000）
    - **MXF 方向濾網**：08:46–08:59 偏多才交易
    - **進場**：開盤買入（漲停略過）
    - **停利**：開盤價 + N tick（台股 tick 制度）
    - **停損**：當日收盤市價平（純時間停損）
    - **手續費**：1.6折 + 當沖稅減半（買 0.0228% + 賣 0.1728%）
    """)

    st.markdown("""
    <div style="background:#1c2128;border:1px solid #f0883e;border-radius:8px;padding:12px 16px;margin:8px 0;">
    ⚠️ <strong>注意：本策略使用日頻最高價判斷 TP，為全天上限估計（過度樂觀）。</strong><br>
    實際早盤 5–15 分鐘命中率更低。後續將以 1 分鐘 K 棒重跑，得到更精確結果。
    </div>
    """, unsafe_allow_html=True)

    rdf_c = load_stock_scalp()
    if rdf_c.empty:
        st.info("回測資料尚未載入")
    else:
        # ── KPI ──
        ticks = sorted(rdf_c["tick"].unique().tolist()) if "tick" in rdf_c.columns else []
        total_trades = len(rdf_c) // max(len(ticks), 1)
        trade_days   = rdf_c["date"].nunique() if "date" in rdf_c.columns else 0
        stocks       = rdf_c["stock"].nunique() if "stock" in rdf_c.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("回測交易日", str(trade_days), "", ""), unsafe_allow_html=True)
        c2.markdown(kpi_card("選股池", f"{stocks} 檔", "曾出現過", ""), unsafe_allow_html=True)
        c3.markdown(kpi_card("Tick 目標", str(len(ticks)), "1~8 tick", ""), unsafe_allow_html=True)
        c4.markdown(kpi_card("資料說明", "日頻上限", "非精確", "neu"), unsafe_allow_html=True)

        # ── Tick 目標彙總 ──
        st.markdown('<div class="sec-header">各 Tick 目標彙總（日頻 TP 上限）</div>', unsafe_allow_html=True)
        summary_rows = []
        for n in ticks:
            t = rdf_c[rdf_c["tick"] == n]
            if t.empty:
                continue
            wins  = (t["pnl"] > 0).sum()
            tp    = (t["result"] == "TP").sum()
            total = len(t)
            std   = t["pnl"].std()
            sharpe = t["pnl"].mean() / std * np.sqrt(252) if std > 0 else 0
            summary_rows.append({
                "tick 目標": n,
                "筆數":       total,
                "TP 率":      f"{tp/total*100:.1f}%",
                "勝率":       f"{wins/total*100:.1f}%",
                "總損益(元)": int(t["pnl"].sum()),
                "平均損益":   round(t["pnl"].mean(), 0),
                "Sharpe":     round(sharpe, 2),
            })

        if summary_rows:
            sdf = pd.DataFrame(summary_rows)
            sdf["總損益(元)"] = sdf["總損益(元)"].apply(fmt_pnl)
            styled_c = sdf.style \
                .map(color_sharpe, subset=["Sharpe"]) \
                .map(color_wr, subset=["勝率"]) \
                .map(color_pnl, subset=["總損益(元)"])
            st.dataframe(styled_c, use_container_width=True, hide_index=True)

        # ── 月份穩定性（最佳 tick）──
        if summary_rows:
            best_tick_row = max(summary_rows, key=lambda r: float(r["Sharpe"]))
            best_tick = best_tick_row["tick 目標"]
            with st.expander(f"月份穩定性（{best_tick} tick，展開）"):
                t_best = rdf_c[rdf_c["tick"] == best_tick]
                if "month" in t_best.columns:
                    rows_m = []
                    for month, mdf in t_best.groupby("month"):
                        wins_m = (mdf["pnl"] > 0).sum()
                        tp_m   = (mdf["result"] == "TP").sum()
                        rows_m.append({
                            "月份": month,
                            "筆數": len(mdf),
                            "TP 率": f"{tp_m/len(mdf)*100:.1f}%",
                            "勝率": f"{wins_m/len(mdf)*100:.1f}%",
                            "損益(元)": int(mdf["pnl"].sum()),
                        })
                    mdf_show = pd.DataFrame(rows_m)
                    mdf_show["損益(元)"] = mdf_show["損益(元)"].apply(fmt_pnl)
                    st.dataframe(mdf_show.style.map(color_pnl, subset=["損益(元)"]),
                                 use_container_width=True, hide_index=True)

        # ── 個股表現 ──
        st.markdown('<div class="sec-header">個股表現（最佳 Tick）</div>', unsafe_allow_html=True)
        if summary_rows:
            t_best = rdf_c[rdf_c["tick"] == best_tick]
            stock_perf = (t_best.groupby("stock")["pnl"]
                          .agg(["sum", "count", "mean"])
                          .rename(columns={"sum": "總損益", "count": "筆數", "mean": "平均損益"})
                          .sort_values("總損益", ascending=False))
            stock_perf["總損益"]   = stock_perf["總損益"].astype(int).apply(fmt_pnl)
            stock_perf["平均損益"] = stock_perf["平均損益"].round(0)
            st.dataframe(
                stock_perf.head(20).style.map(color_pnl, subset=["總損益"]),
                use_container_width=True
            )

        st.markdown("""
        <div style="background:#1c2128;border:1px solid #388bfd;border-radius:8px;padding:10px 16px;margin:12px 0;font-size:13px;color:#8b949e;">
        🔄 <strong>下一步：</strong>VM 正在拉取 45 檔大型股的 1 分鐘 K 棒（Shioaji）。
        完成後將重跑策略 C，改用 5/10/15 分鐘窗口判斷 TP，結果將更接近實際。
        </div>
        """, unsafe_allow_html=True)

    st.caption("完整明細可下載：results_stock_scalp.csv")
    if not rdf_c.empty:
        st.download_button(
            "下載策略 C 明細 CSV",
            rdf_c.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "results_stock_scalp.csv",
            "text/csv",
        )
