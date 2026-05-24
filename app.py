"""
TMF 微型台指期貨 — 回測結果儀表板
策略 A：波動突破
策略 B：固定時間進多 + 高掛 Limit  ★ 主策略
策略 C：前日強勢大型股（1-min K 棒）
策略 D：台積電單股
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

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stHeader"]           { background: transparent; }

.kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 4px 0;
    text-align: center;
}
.kpi-label { color: #8b949e; font-size: 12px; margin-bottom: 4px; letter-spacing: .5px; }
.kpi-value { color: #e6edf3; font-size: 26px; font-weight: 700; }
.kpi-sub   { color: #8b949e; font-size: 11px; margin-top: 3px; }

.pos  { color: #3fb950 !important; }
.neg  { color: #f85149 !important; }
.neu  { color: #d29922 !important; }

.sec-header {
    border-left: 4px solid #388bfd;
    padding-left: 10px;
    color: #e6edf3;
    font-size: 15px;
    font-weight: 600;
    margin: 20px 0 8px;
}

.verdict-ok  { background:#1c2128; border:1px solid #3fb950; border-radius:8px; padding:10px 14px; margin:8px 0; }
.verdict-warn{ background:#1c2128; border:1px solid #d29922; border-radius:8px; padding:10px 14px; margin:8px 0; }
.verdict-bad { background:#1c2128; border:1px solid #f85149; border-radius:8px; padding:10px 14px; margin:8px 0; }

[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── data loaders ─────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(ttl=300)
def load_breakout():
    df = _read_csv("results_breakout.csv")
    trades = _read_csv("results_breakout_trades.csv")
    return df, trades


@st.cache_data(ttl=300)
def load_scalp():
    df = _read_csv("results_scalp.csv")
    trades = _read_csv("results_scalp_trades.csv")
    return df, trades


@st.cache_data(ttl=300)
def load_tsmc():
    return _read_csv("results_tsmc.csv")


@st.cache_data(ttl=300)
def load_stock_scalp():
    df = _read_csv("results_stock_scalp_1min.csv")
    if df.empty:
        df = _read_csv("results_stock_scalp.csv")
    is_1min = "time_window" in df.columns
    return df, is_1min


def _read_csv(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


# ── ui helpers ───────────────────────────────────────────────────────

def kpi_card(label, value, sub="", cls=""):
    cc = f" {cls}" if cls else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value{cc}">{value}</div>'
            + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
            + "</div>")


def sec(title):
    st.markdown(f'<div class="sec-header">{title}</div>', unsafe_allow_html=True)


def color_pnl(val):
    try:
        v = float(str(val).replace(",", "").replace("+", ""))
        if v > 0:  return "color:#3fb950;font-weight:600"
        if v < 0:  return "color:#f85149;font-weight:600"
    except Exception:
        pass
    return ""


def color_sharpe(val):
    try:
        v = float(val)
        if v >= 5:  return "color:#3fb950;font-weight:700"
        if v >= 2:  return "color:#d29922;font-weight:600"
        if v < 0:   return "color:#f85149"
    except Exception:
        pass
    return ""


def color_wr(val):
    try:
        v = float(str(val).replace("%", ""))
        if v >= 60: return "color:#3fb950"
        if v >= 50: return "color:#d29922"
        return "color:#f85149"
    except Exception:
        pass
    return ""


def color_pf(val):
    try:
        v = float(val)
        if v >= 2:  return "color:#3fb950;font-weight:600"
        if v >= 1:  return "color:#d29922"
        return "color:#f85149"
    except Exception:
        pass
    return ""


def fmt_pnl(x):
    try:
        v = int(float(x))
        return f"{v:+,}"
    except Exception:
        return x


def fmt_pct(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return x


def apply_style(df_disp, sharpe_col=None, wr_col=None, pnl_cols=None, pf_col=None):
    s = df_disp.style
    if sharpe_col and sharpe_col in df_disp.columns:
        s = s.map(color_sharpe, subset=[sharpe_col])
    if wr_col and wr_col in df_disp.columns:
        s = s.map(color_wr, subset=[wr_col])
    if pf_col and pf_col in df_disp.columns:
        s = s.map(color_pf, subset=[pf_col])
    for col in (pnl_cols or []):
        if col in df_disp.columns:
            s = s.map(color_pnl, subset=[col])
    return s


def cumulative_chart(trades_df: pd.DataFrame, pnl_col: str, date_col: str = "date"):
    if trades_df.empty or pnl_col not in trades_df.columns:
        st.info("無 trade-level 資料")
        return
    tdf = trades_df.sort_values(date_col).copy()
    tdf["累積損益"] = tdf[pnl_col].cumsum()
    tdf["日期"]     = pd.to_datetime(tdf[date_col])
    daily = tdf.groupby("日期")["累積損益"].last().reset_index()
    st.line_chart(daily.set_index("日期"), height=220)


def monthly_table(trades_df: pd.DataFrame, pnl_col: str):
    if trades_df.empty or "month" not in trades_df.columns:
        return pd.DataFrame()
    rows = []
    for month, mdf in trades_df.groupby("month"):
        wins   = (mdf[pnl_col] > 0).sum()
        losses = (mdf[pnl_col] < 0).sum()
        total  = len(mdf)
        win_avg  = mdf[mdf[pnl_col] > 0][pnl_col].mean() if wins else 0
        loss_avg = mdf[mdf[pnl_col] < 0][pnl_col].mean() if losses else 0
        pf = (win_avg * wins) / abs(loss_avg * losses) if losses > 0 and loss_avg != 0 else 99.0
        rows.append({
            "月份":      month,
            "交易日":    total,
            "勝率":      fmt_pct(wins / total * 100),
            "賺賠比":    round(abs(win_avg / loss_avg), 2) if loss_avg != 0 else 99.0,
            "期望值(元)": round(mdf[pnl_col].mean(), 0),
            "損益(元)":  int(mdf[pnl_col].sum()),
        })
    return pd.DataFrame(rows)


def date_sorted_table(trades_df: pd.DataFrame, pnl_col: str, extra_cols: list = None):
    if trades_df.empty:
        return pd.DataFrame()
    cols = ["date", "result", pnl_col] + (extra_cols or [])
    cols = [c for c in cols if c in trades_df.columns]
    tdf = trades_df[cols].sort_values("date").copy()
    tdf[pnl_col] = tdf[pnl_col].apply(fmt_pnl)
    return tdf


def period_stats(df: pd.DataFrame, col: str = "pnl_twd") -> dict | None:
    if df.empty or col not in df.columns:
        return None
    pnl    = df[col].astype(float)
    std    = pnl.std()
    sharpe = pnl.mean() / std * np.sqrt(252) if std > 0 else 0
    wins   = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    w_avg  = pnl[pnl > 0].mean() if wins else 0
    l_avg  = pnl[pnl < 0].mean() if losses else 0
    pf     = (w_avg * wins) / abs(l_avg * losses) if losses > 0 and l_avg != 0 else 99.0
    return dict(
        sharpe    = round(sharpe, 2),
        total_pnl = int(pnl.sum()),
        win_rate  = round(wins / len(df) * 100, 1),
        pf        = round(pf, 2),
        trades    = len(df),
        ev        = round(pnl.mean(), 1),
    )


def sharpe_cls(s):
    if s >= 2:   return "pos"
    if s >= 0:   return "neu"
    return "neg"


# ════════════════════════════════════════════════════════════════════

st.markdown("## 📊 TMF 微型台指期貨 — 回測儀表板")
st.caption("資料期間：2024-01-02 ~ 2026-05-23（708 交易日）｜手續費：NT$40 round-trip（4 點）")

tab_b, tab_a, tab_c, tab_tsmc = st.tabs(["⭐ 策略 B：固定時間進多", "策略 A：波動突破", "策略 C：前日強勢大型股", "策略 D：台積電單股"])


# ════════════════════════════════════════════════════════════════════
# TAB B  ★ 主策略
# ════════════════════════════════════════════════════════════════════
with tab_b:
    sec("策略邏輯")
    st.markdown("""
    - **方向濾網**：08:46 用夜盤（前日 15:00～當日 05:00）；08:47~09:00 用 08:46～進場前盤前走勢
    - **停利**：掛 Limit 在 entry + target 點；**停損**：可選無停損或固定停損
    - **時間停損**：time_limit 根 K 棒後市價平
    - **參數空間**：15 進場時間 × 8 目標 × 6 停損 × 5/6 時限 ≈ 3,600 組
    """)

    rdf_b, trades_b = load_scalp()

    if rdf_b.empty:
        st.info("回測資料尚未載入")
    else:
        best_b  = rdf_b.loc[rdf_b["sharpe"].idxmax()]
        tp_b    = int(best_b["total_pnl"])
        pf_b    = float(best_b.get("profit_factor", 0))
        ev_b    = float(best_b.get("expect_val", best_b.get("avg_pnl", 0)))

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        cards_b = [
            ("Sharpe 值",     f"{best_b['sharpe']:.2f}",
             f"{best_b['trigger']}  tgt={int(best_b['target'])}  stp={int(best_b['stop'])}", "pos"),
            ("總損益",         f"{tp_b:+,} 元",   "最高 Sharpe 組合（2yr）", "pos" if tp_b>0 else "neg"),
            ("TP 率",          fmt_pct(best_b["tp_rate"]),  "最高 Sharpe 組合", "pos"),
            ("賺賠比",         f"{pf_b:.2f}",      "profit factor", "pos" if pf_b>=1 else "neg"),
            ("期望值(元/筆)",  f"{int(ev_b):+,}",  "每筆平均損益", "pos" if ev_b>0 else "neg"),
            ("正 Sharpe 組數", f"{(rdf_b['sharpe']>0).sum()}", f"共 {len(rdf_b)} 組", ""),
        ]
        for col, (lbl, val, sub, cls) in zip([c1,c2,c3,c4,c5,c6], cards_b):
            col.markdown(kpi_card(lbl, val, sub, cls), unsafe_allow_html=True)

        # ── Train / Test Split ──────────────────────────────────────
        if not trades_b.empty and "date" in trades_b.columns:
            trades_b = trades_b.copy()
            trades_b["date"] = pd.to_datetime(trades_b["date"])

            train_b = trades_b[trades_b["date"] < "2025-01-01"].copy()
            test_b  = trades_b[trades_b["date"] >= "2025-01-01"].copy()

            tr = period_stats(train_b, "pnl_twd")
            te = period_stats(test_b,  "pnl_twd")

            st.markdown("---")
            sec("跨期間一致性驗證（最佳組合）")
            st.caption("注意：最佳參數由完整 2 年資料選出，以下為同參數在不同期間的表現一致性驗證（非嚴格 Out-of-Sample）")

            col_tr, col_te = st.columns(2)

            with col_tr:
                st.markdown("##### 訓練期 2024（參數選取基準，~240 交易日）")
                if tr:
                    t1,t2,t3,t4 = st.columns(4)
                    for c, (l,v,s,cls) in zip([t1,t2,t3,t4], [
                        ("Sharpe",   f"{tr['sharpe']:.2f}", f"{tr['trades']} 筆", sharpe_cls(tr['sharpe'])),
                        ("損益",     f"{tr['total_pnl']:+,}", "元", "pos" if tr['total_pnl']>0 else "neg"),
                        ("勝率",     f"{tr['win_rate']:.1f}%", "", "pos" if tr['win_rate']>=60 else "neu"),
                        ("賺賠比",   f"{tr['pf']:.2f}", "", "pos" if tr['pf']>=1 else "neg"),
                    ]):
                        c.markdown(kpi_card(l,v,s,cls), unsafe_allow_html=True)
                    cumulative_chart(train_b, "pnl_twd")

            with col_te:
                st.markdown("##### 測試期 2025~2026（真實驗證，~480 交易日）")
                if te:
                    t1,t2,t3,t4 = st.columns(4)
                    for c, (l,v,s,cls) in zip([t1,t2,t3,t4], [
                        ("Sharpe",   f"{te['sharpe']:.2f}", f"{te['trades']} 筆", sharpe_cls(te['sharpe'])),
                        ("損益",     f"{te['total_pnl']:+,}", "元", "pos" if te['total_pnl']>0 else "neg"),
                        ("勝率",     f"{te['win_rate']:.1f}%", "", "pos" if te['win_rate']>=60 else "neu"),
                        ("賺賠比",   f"{te['pf']:.2f}", "", "pos" if te['pf']>=1 else "neg"),
                    ]):
                        c.markdown(kpi_card(l,v,s,cls), unsafe_allow_html=True)
                    cumulative_chart(test_b, "pnl_twd")

            # Consistency verdict
            if tr and te:
                robust = te["sharpe"] >= 0 and (tr["sharpe"] <= 0 or te["sharpe"] >= tr["sharpe"] * 0.5)
                if robust and te["sharpe"] >= 2:
                    vclass = "verdict-ok"
                    verdict = f"✅ 測試期 Sharpe {te['sharpe']:.2f} 穩健（訓練期 {tr['sharpe']:.2f}），策略跨期間表現一致"
                elif robust:
                    vclass = "verdict-warn"
                    verdict = f"⚠️ 測試期 Sharpe {te['sharpe']:.2f} 為正但偏低（訓練期 {tr['sharpe']:.2f}），建議持續監控"
                else:
                    vclass = "verdict-bad"
                    verdict = f"❌ 測試期 Sharpe {te['sharpe']:.2f} 顯著低於訓練期 {tr['sharpe']:.2f}，有過擬合風險"
                st.markdown(f'<div class="{vclass}" style="color:#e6edf3">{verdict}</div>', unsafe_allow_html=True)

        # ── Monthly Table ───────────────────────────────────────────
        if not trades_b.empty:
            sec("月份穩定性（最佳組合，2024-2026）")
            mth_b = monthly_table(trades_b, "pnl_twd")
            if not mth_b.empty:
                mth_b["損益(元)"] = mth_b["損益(元)"].apply(fmt_pnl)
                st.dataframe(
                    apply_style(mth_b, wr_col="勝率", pf_col="賺賠比", pnl_cols=["損益(元)"]),
                    use_container_width=True, hide_index=True
                )

        # ── Overfitting Risk ────────────────────────────────────────
        sec("過擬合風險評估")
        col_ov1, col_ov2 = st.columns(2)

        with col_ov1:
            st.markdown("**各進場時間 最佳 Sharpe 分布**")
            best_per_trig = (
                rdf_b.loc[rdf_b.groupby("trigger")["sharpe"].idxmax()]
                [["trigger","sharpe"]]
                .sort_values("trigger")
                .set_index("trigger")
            )
            st.bar_chart(best_per_trig, height=220)

            top10_sharpes = rdf_b.nlargest(10, "sharpe")["sharpe"]
            top1_val  = float(best_b["sharpe"])
            top10_med = top10_sharpes.median()
            st.markdown(f"**Top-1 Sharpe**: {top1_val:.2f} ｜ **Top-10 中位 Sharpe**: {top10_med:.2f}")
            if top1_val > top10_med * 1.8:
                st.warning("Top-1 遠高於 Top-10 中位，存在精選效應風險")
            else:
                st.success("Top-1 與 Top-10 中位接近，整體參數穩健")

        with col_ov2:
            st.markdown(f"**{best_b['trigger']} × stop={int(best_b['stop'])} × 各目標點 Sharpe**")
            trig_sub = (
                rdf_b[
                    (rdf_b["trigger"] == best_b["trigger"]) &
                    (rdf_b["stop"] == best_b["stop"])
                ]
                .groupby("target")["sharpe"].max()
                .reset_index()
                .set_index("target")
            )
            st.bar_chart(trig_sub, height=220)

            pos_tgt   = (trig_sub["sharpe"] > 0).sum()
            total_tgt = len(trig_sub)
            stability = pos_tgt / total_tgt if total_tgt > 0 else 0
            if stability >= 0.7:
                st.success(f"止盈目標 {pos_tgt}/{total_tgt} 組 Sharpe > 0 ── 目標參數穩健")
            elif stability >= 0.5:
                st.warning(f"止盈目標 {pos_tgt}/{total_tgt} 組 Sharpe > 0 ── 部分區間不穩定")
            else:
                st.error(f"止盈目標 {pos_tgt}/{total_tgt} 組 Sharpe > 0 ── 參數敏感，謹慎使用")

        # ── Filter + Full parameter table ───────────────────────────
        sec("篩選器")
        cf1, cf2, cf3, cf4 = st.columns(4)
        trigs_b  = ["全部"] + sorted(rdf_b["trigger"].unique().tolist())
        sel_tb   = cf1.selectbox("觸發時間", trigs_b, key="b_trig")
        sel_stp  = cf2.selectbox("停損", ["全部","無停損(0)","有停損"], key="b_stp")
        min_sb   = cf3.slider("最低 Sharpe", -10.0, 15.0, 0.0, 0.5, key="b_shr")
        min_tp_b = cf4.slider("最低 TP 率 (%)", 0, 100, 0, 5, key="b_tp")

        fdf_b = rdf_b.copy()
        if sel_tb != "全部":        fdf_b = fdf_b[fdf_b["trigger"] == sel_tb]
        if sel_stp == "無停損(0)":  fdf_b = fdf_b[fdf_b["stop"] == 0]
        elif sel_stp == "有停損":   fdf_b = fdf_b[fdf_b["stop"] > 0]
        fdf_b = fdf_b[(fdf_b["sharpe"] >= min_sb) & (fdf_b["tp_rate"] >= min_tp_b)]

        COLS_B = ["trigger","target","stop","time_limit","trades",
                  "tp_rate","sl_rate","timeout_rate","win_rate",
                  "total_pnl","avg_pnl","profit_factor","expect_val","sharpe"]
        COLS_B = [c for c in COLS_B if c in fdf_b.columns]

        sec(f"全參數掃描（Sharpe 排序）— {len(fdf_b):,} 組符合")
        disp_b = fdf_b.sort_values("sharpe", ascending=False).head(30)[COLS_B].copy()
        for c in ["total_pnl","avg_pnl","expect_val"]:
            if c in disp_b.columns: disp_b[c] = disp_b[c].apply(fmt_pnl)
        for c in ["tp_rate","sl_rate","timeout_rate","win_rate"]:
            if c in disp_b.columns: disp_b[c] = disp_b[c].apply(fmt_pct)
        disp_b.columns = [c.replace("_"," ").title() for c in disp_b.columns]
        st.dataframe(
            apply_style(disp_b, sharpe_col="Sharpe", wr_col="Win Rate",
                        pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl","Expect Val"]),
            use_container_width=True, hide_index=True, height=480
        )

        # ── Best per trigger time ───────────────────────────────────
        sec("各進場時間最佳組合")
        best_per_time = rdf_b.loc[rdf_b.groupby("trigger")["sharpe"].idxmax()][
            [c for c in COLS_B if c in rdf_b.columns]].copy()
        for c in ["total_pnl","avg_pnl","expect_val"]:
            if c in best_per_time.columns: best_per_time[c] = best_per_time[c].apply(fmt_pnl)
        for c in ["tp_rate","sl_rate","timeout_rate","win_rate"]:
            if c in best_per_time.columns: best_per_time[c] = best_per_time[c].apply(fmt_pct)
        best_per_time = best_per_time.sort_values("trigger")
        best_per_time.columns = [c.replace("_"," ").title() for c in best_per_time.columns]
        st.dataframe(
            apply_style(best_per_time, sharpe_col="Sharpe", wr_col="Win Rate",
                        pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl","Expect Val"]),
            use_container_width=True, hide_index=True
        )

        # ── Stop comparison ─────────────────────────────────────────
        sec("停損對比（深入分析）")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown(f"**{best_b['trigger']} × 各停損設定（target={int(best_b['target'])}，每停損取最佳 tlim）**")
            sub8 = (
                rdf_b[
                    (rdf_b["trigger"] == best_b["trigger"]) &
                    (rdf_b["target"]  == best_b["target"])
                ]
                .sort_values(["stop","sharpe"], ascending=[True, False])
                .drop_duplicates(subset=["stop"])[
                    [c for c in COLS_B if c in rdf_b.columns]
                ].copy()
            )
            for c in ["total_pnl","avg_pnl","expect_val"]:
                if c in sub8.columns: sub8[c] = sub8[c].apply(fmt_pnl)
            for c in ["tp_rate","sl_rate","timeout_rate","win_rate"]:
                if c in sub8.columns: sub8[c] = sub8[c].apply(fmt_pct)
            sub8.columns = [c.replace("_"," ").title() for c in sub8.columns]
            st.dataframe(apply_style(sub8, sharpe_col="Sharpe", wr_col="Win Rate",
                                     pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl"]),
                         use_container_width=True, hide_index=True)

        with col_r:
            st.markdown("**09:00 × 無停損 × 各時間限制**")
            sub9 = rdf_b[(rdf_b["trigger"]=="09:00") & (rdf_b["stop"]==0)].sort_values("time_limit")[
                [c for c in COLS_B if c in rdf_b.columns]].copy()
            for c in ["total_pnl","avg_pnl","expect_val"]:
                if c in sub9.columns: sub9[c] = sub9[c].apply(fmt_pnl)
            for c in ["tp_rate","sl_rate","timeout_rate","win_rate"]:
                if c in sub9.columns: sub9[c] = sub9[c].apply(fmt_pct)
            sub9.columns = [c.replace("_"," ").title() for c in sub9.columns]
            st.dataframe(apply_style(sub9, sharpe_col="Sharpe", wr_col="Win Rate",
                                     pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl"]),
                         use_container_width=True, hide_index=True)

        # ── Trade log ───────────────────────────────────────────────
        if not trades_b.empty:
            with st.expander("逐筆交易紀錄（最佳組合，按日期排序）"):
                dt_b = date_sorted_table(trades_b, "pnl_twd", ["result","pnl_pts"])
                dt_b.columns = [c.replace("_"," ").title() for c in dt_b.columns]
                st.dataframe(dt_b, use_container_width=True, hide_index=True)

        st.caption("⚠️ 夜盤方向濾網為合法前置資訊（前日 15:00～當日 05:00），無 Look-Ahead Bias")


# ════════════════════════════════════════════════════════════════════
# TAB A
# ════════════════════════════════════════════════════════════════════
with tab_a:
    sec("策略邏輯")
    st.markdown("""
    - **觸發**：08:45 / 09:00 / 13:30 / 13:45，K 棒開盤價 ± offset 才進場
    - **停利 / 停損**：進場後 target / stop 點，或 time_limit 根 K 棒到期市價平
    - **參數空間**：3,920 組
    """)

    rdf_a, trades_a = load_breakout()

    if rdf_a.empty:
        st.info("回測資料尚未載入")
    else:
        best_a = rdf_a.loc[rdf_a["sharpe"].idxmax()]
        tp_a   = int(best_a["total_pnl"])
        pf_a   = best_a.get("profit_factor", "—")
        ev_a   = best_a.get("expect_val", best_a.get("avg_pnl", "—"))

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        cards_a = [
            ("Sharpe 值",     f"{best_a['sharpe']:.2f}",     f"{best_a['trigger']}", "pos"),
            ("總損益",         f"{tp_a:+,} 元",                "最高 Sharpe 組合", "pos" if tp_a>0 else "neg"),
            ("勝率",           fmt_pct(best_a["win_rate"]),    "最高 Sharpe 組合", ""),
            ("賺賠比",         f"{pf_a:.2f}",                  "profit factor", "pos" if float(pf_a)>=1 else "neg"),
            ("期望值(元/筆)",  f"{int(float(ev_a)):+,}",       "每筆平均損益", "pos" if float(ev_a)>0 else "neg"),
            ("正 Sharpe 組數", f"{(rdf_a['sharpe']>0).sum()}", f"共 {len(rdf_a)} 組", ""),
        ]
        for col, (lbl, val, sub, cls) in zip([c1,c2,c3,c4,c5,c6], cards_a):
            col.markdown(kpi_card(lbl, val, sub, cls), unsafe_allow_html=True)

        if not trades_a.empty:
            sec("最佳組合累積損益曲線（按日期）")
            cumulative_chart(trades_a, "pnl_twd")

        if not trades_a.empty:
            sec("月份穩定性（最佳組合）")
            mth_a = monthly_table(trades_a, "pnl_twd")
            if not mth_a.empty:
                mth_a["損益(元)"] = mth_a["損益(元)"].apply(fmt_pnl)
                st.dataframe(
                    apply_style(mth_a, wr_col="勝率", pf_col="賺賠比", pnl_cols=["損益(元)"]),
                    use_container_width=True, hide_index=True
                )

        sec("篩選器")
        cf1, cf2, cf3 = st.columns(3)
        triggers_a = ["全部"] + sorted(rdf_a["trigger"].unique().tolist())
        sel_trig_a  = cf1.selectbox("觸發時間", triggers_a, key="a_trig")
        min_shr_a   = cf2.slider("最低 Sharpe", -5.0, 15.0, 0.0, 0.5, key="a_shr")
        min_wr_a    = cf3.slider("最低勝率 (%)", 0, 100, 0, 5, key="a_wr")

        fdf_a = rdf_a.copy()
        if sel_trig_a != "全部":
            fdf_a = fdf_a[fdf_a["trigger"] == sel_trig_a]
        fdf_a = fdf_a[(fdf_a["sharpe"] >= min_shr_a) & (fdf_a["win_rate"] >= min_wr_a)]

        COLS_A = ["trigger","offset","target","stop","time_limit",
                  "trades","win_rate","total_pnl","avg_pnl",
                  "profit_factor","expect_val","sharpe"]
        COLS_A = [c for c in COLS_A if c in fdf_a.columns]

        sec(f"Top 30（Sharpe）— {len(fdf_a):,} 組符合")
        disp_a = fdf_a.sort_values("sharpe", ascending=False).head(30)[COLS_A].copy()
        for c in ["total_pnl","avg_pnl","expect_val"]:
            if c in disp_a.columns: disp_a[c] = disp_a[c].apply(fmt_pnl)
        if "win_rate" in disp_a.columns:
            disp_a["win_rate"] = disp_a["win_rate"].apply(fmt_pct)
        disp_a.columns = [c.replace("_"," ").title() for c in disp_a.columns]
        st.dataframe(
            apply_style(disp_a, sharpe_col="Sharpe", wr_col="Win Rate",
                        pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl","Expect Val"]),
            use_container_width=True, hide_index=True, height=480
        )

        sec("各觸發時間最佳組合")
        best_per_a = rdf_a.loc[rdf_a.groupby("trigger")["sharpe"].idxmax()][COLS_A].copy()
        for c in ["total_pnl","avg_pnl","expect_val"]:
            if c in best_per_a.columns: best_per_a[c] = best_per_a[c].apply(fmt_pnl)
        if "win_rate" in best_per_a.columns:
            best_per_a["win_rate"] = best_per_a["win_rate"].apply(fmt_pct)
        best_per_a.columns = [c.replace("_"," ").title() for c in best_per_a.columns]
        st.dataframe(
            apply_style(best_per_a, sharpe_col="Sharpe", wr_col="Win Rate",
                        pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl","Expect Val"]),
            use_container_width=True, hide_index=True
        )

        sec("各觸發時間最佳總損益")
        chart_a = rdf_a.loc[rdf_a.groupby("trigger")["sharpe"].idxmax()][["trigger","total_pnl"]].set_index("trigger")
        st.bar_chart(chart_a, height=240)

        if not trades_a.empty:
            with st.expander("逐筆交易紀錄（最佳組合，按日期排序）"):
                dt_a = date_sorted_table(trades_a, "pnl_twd", ["direction","pnl_pts"])
                dt_a.columns = [c.replace("_"," ").title() for c in dt_a.columns]
                st.dataframe(dt_a, use_container_width=True, hide_index=True)

        st.caption("⚠️ 參數掃描結果，請注意過擬合風險（3,920 組合）")


# ════════════════════════════════════════════════════════════════════
# TAB C
# ════════════════════════════════════════════════════════════════════
with tab_c:
    sec("策略邏輯")
    st.markdown("""
    - **選股**：前一日漲幅前 30 名（市值 > 500 億，價格 100-200 或 1000-2000）
    - **MXF 方向濾網**：08:46–08:59 偏多才交易
    - **進場**：09:01 開盤第一根 K 棒（漲停略過）
    - **停利**：進場後 N tick，時間窗口（5/10/15 min）內達到即觸發
    - **手續費**：1.6折 + 當沖稅減半（買 0.0228% + 賣 0.1728%）
    """)

    st.markdown("""
    <div style="background:#1c2128;border:1px solid #f85149;border-radius:8px;padding:12px 16px;margin:8px 0;">
    <strong>Gap-Fade 效應：</strong>前日強勢股開盤後持續回落，做多全面虧損。<br>
    <strong>建議改進：反向做空前日強勢股，或做多前日弱勢股（均值回歸）。</strong>
    </div>
    """, unsafe_allow_html=True)

    rdf_c, is_1min = load_stock_scalp()

    if rdf_c.empty:
        st.info("回測資料尚未載入")
    else:
        ticks   = sorted(rdf_c["tick"].unique().tolist()) if "tick" in rdf_c.columns else []
        windows = sorted(rdf_c["time_window"].unique().tolist()) if is_1min and "time_window" in rdf_c.columns else []
        tdays   = rdf_c["date"].nunique() if "date" in rdf_c.columns else 0
        stocks  = rdf_c["stock"].nunique() if "stock" in rdf_c.columns else 0

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(kpi_card("回測交易日", str(tdays), "MXF 偏多日", ""), unsafe_allow_html=True)
        c2.markdown(kpi_card("選股池", f"{stocks} 檔", "曾入選過", ""), unsafe_allow_html=True)
        c3.markdown(kpi_card("Tick 目標", f"{ticks[0]}~{ticks[-1]}" if ticks else "—", "1~8 tick", ""), unsafe_allow_html=True)
        c4.markdown(kpi_card("資料品質", "1-min K 棒" if is_1min else "日頻上限", "", "pos" if is_1min else "neu"), unsafe_allow_html=True)

        if is_1min:
            sec("各 Tick × 時間窗口 表現（1-min K 棒，精確）")
            sel_tw = st.selectbox("時間窗口（分鐘）", ["全部"] + [str(w) for w in windows], key="c_tw")
        else:
            sec("各 Tick 目標彙總（日頻 TP 上限）")
            sel_tw = "全部"

        summary_rows = []
        for n in ticks:
            for tw in (windows if is_1min else [None]):
                mask = rdf_c["tick"] == n
                if is_1min and tw is not None:
                    mask &= rdf_c["time_window"] == tw
                t = rdf_c[mask]
                if t.empty: continue
                wins   = (t["pnl"] > 0).sum()
                losses = (t["pnl"] < 0).sum()
                tp_n   = (t["result"] == "TP").sum()
                total  = len(t)
                std    = t["pnl"].std()
                sharpe = t["pnl"].mean() / std * np.sqrt(252) if std > 0 else 0
                win_avg  = t[t["pnl"]>0]["pnl"].mean() if wins else 0
                loss_avg = t[t["pnl"]<0]["pnl"].mean() if losses else 0
                pf = abs(win_avg * wins / (loss_avg * losses)) if losses > 0 and loss_avg != 0 else 99.0
                row = {
                    "tick 目標": n,
                    "筆數":     total,
                    "TP 率":    fmt_pct(tp_n/total*100),
                    "勝率":     fmt_pct(wins/total*100),
                    "賺賠比":   round(pf, 2),
                    "期望值(元)": round(t["pnl"].mean(), 0),
                    "總損益(元)": int(t["pnl"].sum()),
                    "Sharpe":   round(sharpe, 2),
                }
                if is_1min:
                    row["時間窗口(min)"] = tw
                summary_rows.append(row)

        if summary_rows:
            sdf = pd.DataFrame(summary_rows)
            if sel_tw != "全部" and "時間窗口(min)" in sdf.columns:
                sdf = sdf[sdf["時間窗口(min)"] == int(sel_tw)]
            sdf_d = sdf.copy()
            sdf_d["總損益(元)"] = sdf_d["總損益(元)"].apply(fmt_pnl)
            st.dataframe(
                apply_style(sdf_d, sharpe_col="Sharpe", wr_col="勝率",
                            pf_col="賺賠比", pnl_cols=["總損益(元)"]),
                use_container_width=True, hide_index=True
            )

        if summary_rows and "date" in rdf_c.columns:
            best_row_c  = max(summary_rows, key=lambda r: float(r["Sharpe"]))
            best_tick_c = best_row_c["tick 目標"]
            best_tw_c   = best_row_c.get("時間窗口(min)", None)
            mask_best   = rdf_c["tick"] == best_tick_c
            if is_1min and best_tw_c:
                mask_best &= rdf_c["time_window"] == best_tw_c
            t_best_c = rdf_c[mask_best].copy()

            label_c = f"{best_tick_c} tick" + (f" × {best_tw_c} min" if best_tw_c else "")
            sec(f"最佳組合累積損益曲線（{label_c}，按日期）")
            cumulative_chart(t_best_c, "pnl")

            sec(f"月份穩定性（{label_c}）")
            mth_c = monthly_table(t_best_c, "pnl")
            if not mth_c.empty:
                mth_c["損益(元)"] = mth_c["損益(元)"].apply(fmt_pnl)
                st.dataframe(
                    apply_style(mth_c, wr_col="勝率", pf_col="賺賠比", pnl_cols=["損益(元)"]),
                    use_container_width=True, hide_index=True
                )

            with st.expander("逐筆交易紀錄（最佳組合，按日期排序）"):
                dt_c = date_sorted_table(t_best_c, "pnl", ["stock","result","tick"])
                dt_c.columns = [c.replace("_"," ").title() for c in dt_c.columns]
                st.dataframe(dt_c, use_container_width=True, hide_index=True)

        sec("Gap-Fade 解析")
        col_ga, col_gb = st.columns(2)
        with col_ga:
            st.markdown("""
            **為什麼全面虧損？**
            - 前日強勢股收盤前已反映漲幅，隔日開盤有溢價 gap
            - 開盤後 5–15 分鐘主力出貨，散戶追高被套
            - 手續費 0.2% round-trip，小 tick 獲利空間極小
            - TIMEOUT 平均虧損遠大於 TP 獲利（~6,800 元 vs ~200 元）
            """)
        with col_gb:
            st.markdown("""
            **建議改進方向**
            1. **做空**前日強勢股（先賣後買當沖，需現股當沖資格）
            2. **做多**前日弱勢股（均值回歸）
            3. **等回落**後進場（09:10–09:20 支撐確認）
            4. **換因子**：法人買超、量能異常、融券高檔等
            """)

        st.download_button(
            "下載策略 C 明細 CSV",
            rdf_c.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "results_stock_scalp_1min.csv" if is_1min else "results_stock_scalp.csv",
            "text/csv",
        )


# ════════════════════════════════════════════════════════════════════
# TAB D：台積電單股
# ════════════════════════════════════════════════════════════════════
with tab_tsmc:
    sec("策略邏輯")
    st.markdown("""
    - **選股**：只做台積電（2330），不篩選前日強弱
    - **進場條件**：MXF 08:46–08:59 偏多才進場
    - **進場**：09:01 開盤第一根 K 棒（漲停略過）
    - **停利**：N tick，時間窗口（5/10/15/20/30 min）內達到即觸發
    - **停損**：時間到市價平（純時間停損）
    - **資料期間**：2024-01-02 ~ 2026-05-22（576 個交易日）
    - **手續費**：1.6折 + 當沖稅減半
    """)

    st.markdown("""
    <div style="background:#1c2128;border:1px solid #f0883e;border-radius:8px;padding:12px 16px;margin:8px 0;">
    ⚠️ <strong>手續費陷阱：</strong>台積電在 800 元時，1 tick TP = 1,000 元，但手續費高達 ~1,565 元，連 TP 也虧損。<br>
    只有台積電在 <strong>1,000 元以上</strong>（tick=5元）才有足夠利潤空間。
    </div>
    """, unsafe_allow_html=True)

    rdf_t = load_tsmc()

    if rdf_t.empty:
        st.info("回測資料尚未載入")
    else:
        ticks_t   = sorted(rdf_t["tick"].unique().tolist()) if "tick" in rdf_t.columns else []
        windows_t = sorted(rdf_t["time_window"].unique().tolist()) if "time_window" in rdf_t.columns else []
        tdays_t   = rdf_t["date"].nunique() if "date" in rdf_t.columns else 0

        summary_t = []
        for n in ticks_t:
            for tw in windows_t:
                t = rdf_t[(rdf_t["tick"] == n) & (rdf_t["time_window"] == tw)]
                if t.empty: continue
                wins   = (t["pnl"] > 0).sum()
                losses = (t["pnl"] < 0).sum()
                tp_n   = (t["result"] == "TP").sum()
                total  = len(t)
                std    = t["pnl"].std()
                sharpe = t["pnl"].mean() / std * np.sqrt(252) if std > 0 else 0
                win_avg  = t[t["pnl"] > 0]["pnl"].mean() if wins else 0
                loss_avg = t[t["pnl"] < 0]["pnl"].mean() if losses else 0
                pf = abs(win_avg * wins / (loss_avg * losses)) if losses > 0 and loss_avg != 0 else 99.0
                summary_t.append({
                    "tick": n, "時間窗口(min)": tw, "交易日": total,
                    "TP 率":   f"{tp_n/total*100:.1f}%",
                    "勝率":    f"{wins/total*100:.1f}%",
                    "賺賠比":  round(pf, 2),
                    "Sharpe":  round(sharpe, 2),
                    "期望值(元)": round(t["pnl"].mean(), 0),
                    "總損益(元)": int(t["pnl"].sum()),
                })

        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi_card("回測交易日", str(tdays_t), "MXF 偏多", ""), unsafe_allow_html=True)
        best_t = max(summary_t, key=lambda r: float(r["Sharpe"])) if summary_t else {}
        c2.markdown(kpi_card("最佳 Sharpe", f"{best_t.get('Sharpe', '—'):.2f}" if best_t else "—",
                             f"{best_t.get('tick','?')} tick × {best_t.get('時間窗口(min)','?')} min" if best_t else "",
                             "pos" if best_t and float(best_t.get("Sharpe","-99")) > 0 else "neg"),
                    unsafe_allow_html=True)
        c3.markdown(kpi_card("價格範圍", "574 ~ 2,340 元", "tick=1(≤1000) / tick=5(>1000)", "neu"),
                    unsafe_allow_html=True)

        sec("各 Tick × 時間窗口 表現")
        sel_tw_t = st.selectbox("時間窗口（分鐘）", ["全部"] + [str(w) for w in windows_t], key="t_tw")
        if summary_t:
            sdf_t = pd.DataFrame(summary_t)
            if sel_tw_t != "全部":
                sdf_t = sdf_t[sdf_t["時間窗口(min)"] == int(sel_tw_t)]
            sdf_t_d = sdf_t.copy()
            sdf_t_d["總損益(元)"] = sdf_t_d["總損益(元)"].apply(fmt_pnl)
            st.dataframe(
                apply_style(sdf_t_d, sharpe_col="Sharpe", wr_col="勝率",
                            pf_col="賺賠比", pnl_cols=["總損益(元)"]),
                use_container_width=True, hide_index=True
            )

        if best_t and "date" in rdf_t.columns:
            t_best_t = rdf_t[(rdf_t["tick"] == best_t["tick"]) &
                             (rdf_t["time_window"] == best_t["時間窗口(min)"])].copy()
            label_t = f"{best_t['tick']} tick × {best_t['時間窗口(min)']} min"
            sec(f"累積損益曲線（{label_t}，按日期）")
            cumulative_chart(t_best_t, "pnl")

            sec(f"月份穩定性（{label_t}）")
            mth_t = monthly_table(t_best_t, "pnl")
            if not mth_t.empty:
                mth_t["損益(元)"] = mth_t["損益(元)"].apply(fmt_pnl)
                st.dataframe(
                    apply_style(mth_t, wr_col="勝率", pf_col="賺賠比", pnl_cols=["損益(元)"]),
                    use_container_width=True, hide_index=True
                )

        sec("分析：為何台積電表現有明顯週期性？")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            **虧損期（2024 上半年）**
            - 台積電 ~600-850 元，tick = **1 元**
            - 手續費 = 0.1956% × 800 × 1000 ≈ **1,565 元**
            - 1 tick TP = 1 × 1000 = **1,000 元**
            - TP 也虧 565 元，完全無利可圖

            **盈利期（2024 Q4 ~ 2025 Q1）**
            - 台積電 ~1,100-1,300 元，tick = **5 元**
            - 手續費 ≈ **2,347 元**
            - 1 tick TP = 5 × 1000 = **5,000 元**
            - 淨 TP = 2,653 元，空間充足
            """)
        with col_t2:
            st.markdown("""
            **改進方向**
            1. **只在台積電 >1,000 元時啟動策略**（動態開關）
            2. **增加趨勢濾網**：台積電高於 20MA 才做多
            3. **調大 tick 目標**：≥3 tick，確保扣除手續費後仍有獲利
            4. **加入前日強弱**：台積電前日也要收漲才進場
            5. **組合下單**：與 MXF 策略對沖，降低整體波動
            """)

        with st.expander("逐筆交易紀錄（最佳組合，按日期排序）"):
            if best_t:
                t_best_t = rdf_t[(rdf_t["tick"] == best_t["tick"]) &
                                 (rdf_t["time_window"] == best_t["時間窗口(min)"])].copy()
                dt_t = date_sorted_table(t_best_t, "pnl", ["result","entry","tick"])
                dt_t.columns = [c.replace("_"," ").title() for c in dt_t.columns]
                st.dataframe(dt_t, use_container_width=True, hide_index=True)

        st.download_button(
            "下載台積電明細 CSV",
            rdf_t.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "results_tsmc.csv", "text/csv",
        )
