"""
TMF 微型台指期貨 — 回測結果儀表板
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

.rec-card {
    background: #161b22;
    border: 1px solid #388bfd;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
}
.rec-title { color: #388bfd; font-size: 13px; font-weight: 700; margin-bottom: 8px; }
.rec-row   { color: #e6edf3; font-size: 14px; margin: 3px 0; }
.rec-label { color: #8b949e; font-size: 12px; margin-right: 6px; }

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

.verdict-ok  { background:#1c2128; border:1px solid #3fb950; border-radius:8px; padding:10px 14px; margin:8px 0; color:#e6edf3; }
.verdict-warn{ background:#1c2128; border:1px solid #d29922; border-radius:8px; padding:10px 14px; margin:8px 0; color:#e6edf3; }
.verdict-bad { background:#1c2128; border:1px solid #f85149; border-radius:8px; padding:10px 14px; margin:8px 0; color:#e6edf3; }

[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── data loaders ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"


def _read_csv(name):
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(ttl=300)
def load_breakout():
    return _read_csv("results_breakout.csv"), _read_csv("results_breakout_trades.csv")


@st.cache_data(ttl=300)
def load_scalp():
    return _read_csv("results_scalp.csv"), _read_csv("results_scalp_trades.csv")


@st.cache_data(ttl=300)
def load_exit():
    return _read_csv("results_scalp_exit.csv"), _read_csv("results_scalp_exit_trades.csv")


@st.cache_data(ttl=300)
def load_tsmc():
    return _read_csv("results_tsmc.csv")


@st.cache_data(ttl=300)
def load_stock_scalp():
    df = _read_csv("results_stock_scalp_1min.csv")
    if df.empty:
        df = _read_csv("results_stock_scalp.csv")
    return df, "time_window" in df.columns


# ── ui helpers ────────────────────────────────────────────────────────────────

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
        if v > 0: return "color:#3fb950;font-weight:600"
        if v < 0: return "color:#f85149;font-weight:600"
    except Exception: pass
    return ""


def color_sharpe(val):
    try:
        v = float(val)
        if v >= 5: return "color:#3fb950;font-weight:700"
        if v >= 2: return "color:#d29922;font-weight:600"
        if v < 0:  return "color:#f85149"
    except Exception: pass
    return ""


def color_wr(val):
    try:
        v = float(str(val).replace("%", ""))
        if v >= 60: return "color:#3fb950"
        if v >= 50: return "color:#d29922"
        return "color:#f85149"
    except Exception: pass
    return ""


def color_pf(val):
    try:
        v = float(val)
        if v >= 2: return "color:#3fb950;font-weight:600"
        if v >= 1: return "color:#d29922"
        return "color:#f85149"
    except Exception: pass
    return ""


def fmt_pnl(x):
    try: return f"{int(float(x)):+,}"
    except Exception: return x


def fmt_pct(x):
    try: return f"{float(x):.1f}%"
    except Exception: return x


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


def cumulative_chart(trades_df, pnl_col, date_col="date", height=220):
    if trades_df.empty or pnl_col not in trades_df.columns:
        st.info("無 trade-level 資料")
        return
    tdf = trades_df.sort_values(date_col).copy()
    tdf["累積損益"] = tdf[pnl_col].cumsum()
    tdf["日期"]     = pd.to_datetime(tdf[date_col])
    daily = tdf.groupby("日期")["累積損益"].last().reset_index()
    st.line_chart(daily.set_index("日期"), height=height)


def monthly_table(trades_df, pnl_col):
    if trades_df.empty or "month" not in trades_df.columns:
        return pd.DataFrame()
    rows = []
    for month, mdf in trades_df.groupby("month"):
        wins  = (mdf[pnl_col] > 0).sum()
        loss  = (mdf[pnl_col] < 0).sum()
        total = len(mdf)
        wa = mdf[mdf[pnl_col] > 0][pnl_col].mean() if wins else 0
        la = mdf[mdf[pnl_col] < 0][pnl_col].mean() if loss else 0
        pf = (wa * wins) / abs(la * loss) if loss > 0 and la != 0 else 99.0
        rows.append({
            "月份": month, "交易日": total,
            "勝率": fmt_pct(wins / total * 100),
            "賺賠比": round(abs(wa / la), 2) if la != 0 else 99.0,
            "期望值(元)": round(mdf[pnl_col].mean(), 0),
            "損益(元)": int(mdf[pnl_col].sum()),
        })
    return pd.DataFrame(rows)


def date_sorted_table(trades_df, pnl_col, extra_cols=None):
    if trades_df.empty:
        return pd.DataFrame()
    cols = list(dict.fromkeys(["date", "result", pnl_col] + (extra_cols or [])))
    cols = [c for c in cols if c in trades_df.columns]
    tdf  = trades_df[cols].sort_values("date").copy()
    tdf[pnl_col] = tdf[pnl_col].apply(fmt_pnl)
    return tdf


def period_stats(df, col):
    if df.empty or col not in df.columns:
        return None
    pnl  = df[col].astype(float)
    std  = pnl.std()
    sharpe = pnl.mean() / std * np.sqrt(252) if std > 0 else 0
    wins = (pnl > 0).sum()
    loss = (pnl < 0).sum()
    wa   = pnl[pnl > 0].mean() if wins else 0
    la   = pnl[pnl < 0].mean() if loss else 0
    pf   = (wa * wins) / abs(la * loss) if loss > 0 and la != 0 else 99.0
    return dict(sharpe=round(sharpe, 2), total_pnl=int(pnl.sum()),
                win_rate=round(wins / len(df) * 100, 1), pf=round(pf, 2),
                trades=len(df), ev=round(pnl.mean(), 1))


def sharpe_cls(s):
    if s >= 2: return "pos"
    if s >= 0: return "neu"
    return "neg"


# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 TMF 微型台指期貨 — 回測儀表板")
st.caption("資料期間：2024-01-02 ~ 2026-05-23（708 交易日）｜手續費：NT$40 round-trip（4 點）")

tab_ov, tab_b_entry, tab_b_exit, tab_a, tab_cd = st.tabs([
    "📋 總覽 & 結論",
    "⭐ 策略B：進場優化",
    "🚪 策略B：出場優化",
    "策略A：波動突破",
    "策略C/D：股票策略",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — 總覽 & 結論
# ════════════════════════════════════════════════════════════════════════════
with tab_ov:
    rdf_b, trades_b = load_scalp()
    rdf_a, trades_a = load_breakout()
    rdf_e, _        = load_exit()

    # ── 各策略比較表 ──────────────────────────────────────────────
    sec("各策略回測結果比較")

    overview_rows = []

    if not rdf_b.empty:
        best_b = rdf_b.loc[rdf_b["sharpe"].idxmax()]
        overview_rows.append({
            "策略": "B 固定時間進多", "狀態": "✅ 使用",
            "觸發": f"{best_b['trigger']} 夜盤方向",
            "最佳參數": f"tgt={int(best_b['target'])} stp={int(best_b['stop'])}",
            "Sharpe": round(best_b["sharpe"], 2),
            "勝率": fmt_pct(best_b["win_rate"]),
            "期望值(元)": round(float(best_b.get("expect_val", best_b.get("avg_pnl", 0))), 0),
            "備註": "測試期表現優於訓練期",
        })
    if not rdf_a.empty:
        best_a = rdf_a.loc[rdf_a["sharpe"].idxmax()]
        overview_rows.append({
            "策略": "A 波動突破", "狀態": "🟡 觀察",
            "觸發": f"{best_a['trigger']}",
            "最佳參數": f"offset={int(best_a['offset'])} tgt={int(best_a['target'])}",
            "Sharpe": round(best_a["sharpe"], 2),
            "勝率": fmt_pct(best_a["win_rate"]),
            "期望值(元)": round(float(best_a.get("expect_val", best_a.get("avg_pnl", 0))), 0),
            "備註": "高 EV，需訓練/測試驗證",
        })
    overview_rows.append({
        "策略": "C 前日強勢股", "狀態": "❌ 不使用",
        "觸發": "09:01 開盤",
        "最佳參數": "—",
        "Sharpe": "全負",
        "勝率": "—",
        "期望值(元)": "—",
        "備註": "Gap-Fade 效應，開盤後持續回落",
    })
    overview_rows.append({
        "策略": "D 台積電單股", "狀態": "❌ 不使用",
        "觸發": "09:01 開盤",
        "最佳參數": "—",
        "Sharpe": "全負",
        "勝率": "—",
        "期望值(元)": "—",
        "備註": "手續費 > 1tick TP（<1000元時）",
    })

    if overview_rows:
        ov_df = pd.DataFrame(overview_rows)
        st.dataframe(ov_df, use_container_width=True, hide_index=True)

    # ── 策略B 建議執行參數 ────────────────────────────────────────
    st.markdown("---")
    sec("策略B 建議執行參數（目前最優解）")

    if not rdf_b.empty:
        best_b = rdf_b.loc[rdf_b["sharpe"].idxmax()]
        pf_b   = float(best_b.get("profit_factor", 0))
        ev_b   = float(best_b.get("expect_val", best_b.get("avg_pnl", 0)))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="rec-card">
            <div class="rec-title">📥 進場條件</div>
            <div class="rec-row"><span class="rec-label">時間</span>08:46（夜盤收盤後）</div>
            <div class="rec-row"><span class="rec-label">方向</span>夜盤偏多（前日15:00~當日05:00）</div>
            <div class="rec-row"><span class="rec-label">進場</span>08:46 開盤價買進</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="rec-card">
            <div class="rec-title">🎯 出場條件</div>
            <div class="rec-row"><span class="rec-label">停利</span>{int(best_b['target'])} pts（限價單）</div>
            <div class="rec-row"><span class="rec-label">停損</span>{int(best_b['stop'])} pts</div>
            <div class="rec-row"><span class="rec-label">時間限制</span>不影響（3~20 bars 結果相同）</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="rec-card">
            <div class="rec-title">📊 預期績效</div>
            <div class="rec-row"><span class="rec-label">Sharpe</span>{best_b['sharpe']:.2f}</div>
            <div class="rec-row"><span class="rec-label">勝率</span>{best_b['win_rate']:.1f}%（TP 率 {best_b['tp_rate']:.1f}%）</div>
            <div class="rec-row"><span class="rec-label">期望值</span>{int(ev_b):+,} 元 / 筆</div>
            </div>
            """, unsafe_allow_html=True)

        # 訓練/測試 Sharpe 比較
        if not trades_b.empty and "date" in trades_b.columns:
            trades_b2 = trades_b.copy()
            trades_b2["date"] = pd.to_datetime(trades_b2["date"])
            tr = period_stats(trades_b2[trades_b2["date"] < "2025-01-01"], "pnl_twd")
            te = period_stats(trades_b2[trades_b2["date"] >= "2025-01-01"], "pnl_twd")
            if tr and te:
                is_robust = te["sharpe"] >= 0 and te["sharpe"] >= tr["sharpe"] * 0.5
                vclass    = "verdict-ok" if te["sharpe"] >= 2 else ("verdict-warn" if is_robust else "verdict-bad")
                verdict   = (f"✅ 跨期間驗證通過：訓練期 Sharpe {tr['sharpe']:.2f}（2024）→ 測試期 Sharpe {te['sharpe']:.2f}（2025-2026），策略穩健"
                             if is_robust and te["sharpe"] >= 2
                             else f"⚠️ 訓練期 Sharpe {tr['sharpe']:.2f} → 測試期 Sharpe {te['sharpe']:.2f}，請持續監控")
                st.markdown(f'<div class="{vclass}">{verdict}</div>', unsafe_allow_html=True)

    # ── 過擬合風險燈號 ──────────────────────────────────────────────
    st.markdown("---")
    sec("過擬合風險燈號")
    r1, r2, r3, r4 = st.columns(4)

    risk_items = [
        ("策略B 進場", "🟢 低", "測試期優於訓練期，月份穩定性高", "pos"),
        ("策略A 突破", "🟡 中", "參數空間大（3,920 組），需訓練/測試拆分確認", "neu"),
        ("策略B 出場", "⬜ 待測", "出場優化回測進行中", ""),
        ("策略C/D 股票", "🔴 已排除", "Gap-Fade 效應 / 手續費陷阱，不採用", "neg"),
    ]
    for col, (name, risk, desc, cls) in zip([r1, r2, r3, r4], risk_items):
        col.markdown(kpi_card(name, risk, desc, cls), unsafe_allow_html=True)

    # ── 優化待辦清單 ──────────────────────────────────────────────
    st.markdown("---")
    sec("優化路線圖")
    col_done, col_todo, col_idea = st.columns(3)
    with col_done:
        st.markdown("""
        **✅ 已完成**
        - 策略B 進場時間掃描（08:46~09:00 全掃）
        - 2 年資料回測（708 交易日）
        - 訓練/測試分割驗證
        - 策略A 波動突破參數優化
        - 過擬合風險評估框架
        """)
    with col_todo:
        st.markdown("""
        **🔄 進行中**
        - 策略B 出場方式回測
          - Full TP vs Partial TP vs Trailing
        """)
    with col_idea:
        st.markdown("""
        **💡 下一步**
        - 09:00 觸發能否與 08:46 組合下單？
        - 加入做空方向（夜盤偏空時做空）
        - 策略A 訓練/測試拆分驗證
        - 盤中加碼邏輯（移動止損鎖利）
        - 正式上線監控（GCP VM + LINE 通知）
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — 策略B 進場優化
# ════════════════════════════════════════════════════════════════════════════
with tab_b_entry:
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
        best_b = rdf_b.loc[rdf_b["sharpe"].idxmax()]
        tp_b   = int(best_b["total_pnl"])
        pf_b   = float(best_b.get("profit_factor", 0))
        ev_b   = float(best_b.get("expect_val", best_b.get("avg_pnl", 0)))

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        cards_b = [
            ("Sharpe 值",    f"{best_b['sharpe']:.2f}",
             f"{best_b['trigger']}  tgt={int(best_b['target'])}  stp={int(best_b['stop'])}", "pos"),
            ("總損益",        f"{tp_b:+,} 元",   "最高 Sharpe 組合（2yr）", "pos" if tp_b>0 else "neg"),
            ("TP 率",         fmt_pct(best_b["tp_rate"]),  "最高 Sharpe 組合", "pos"),
            ("賺賠比",        f"{pf_b:.2f}",      "profit factor", "pos" if pf_b>=1 else "neg"),
            ("期望值(元/筆)", f"{int(ev_b):+,}",  "每筆平均損益", "pos" if ev_b>0 else "neg"),
            ("正Sharpe 組數", f"{(rdf_b['sharpe']>0).sum()}", f"共 {len(rdf_b)} 組", ""),
        ]
        for col, (lbl, val, sub, cls) in zip([c1,c2,c3,c4,c5,c6], cards_b):
            col.markdown(kpi_card(lbl, val, sub, cls), unsafe_allow_html=True)

        # Train / Test
        if not trades_b.empty and "date" in trades_b.columns:
            trades_b = trades_b.copy()
            trades_b["date"] = pd.to_datetime(trades_b["date"])
            train_b = trades_b[trades_b["date"] < "2025-01-01"].copy()
            test_b  = trades_b[trades_b["date"] >= "2025-01-01"].copy()
            tr = period_stats(train_b, "pnl_twd")
            te = period_stats(test_b,  "pnl_twd")

            st.markdown("---")
            sec("跨期間一致性驗證（最佳組合）")
            st.caption("最佳參數由完整 2 年資料選出，以下為同參數在不同期間的一致性驗證（非嚴格 Out-of-Sample）")
            col_tr, col_te = st.columns(2)
            with col_tr:
                st.markdown("##### 訓練期 2024（~240 交易日）")
                if tr:
                    t1,t2,t3,t4 = st.columns(4)
                    for c,(l,v,s,cls) in zip([t1,t2,t3,t4],[
                        ("Sharpe",f"{tr['sharpe']:.2f}",f"{tr['trades']} 筆",sharpe_cls(tr['sharpe'])),
                        ("損益",f"{tr['total_pnl']:+,}","元","pos" if tr['total_pnl']>0 else "neg"),
                        ("勝率",f"{tr['win_rate']:.1f}%","","pos" if tr['win_rate']>=60 else "neu"),
                        ("賺賠比",f"{tr['pf']:.2f}","","pos" if tr['pf']>=1 else "neg"),
                    ]):
                        c.markdown(kpi_card(l,v,s,cls), unsafe_allow_html=True)
                    cumulative_chart(train_b, "pnl_twd")
            with col_te:
                st.markdown("##### 測試期 2025~2026（~480 交易日）")
                if te:
                    t1,t2,t3,t4 = st.columns(4)
                    for c,(l,v,s,cls) in zip([t1,t2,t3,t4],[
                        ("Sharpe",f"{te['sharpe']:.2f}",f"{te['trades']} 筆",sharpe_cls(te['sharpe'])),
                        ("損益",f"{te['total_pnl']:+,}","元","pos" if te['total_pnl']>0 else "neg"),
                        ("勝率",f"{te['win_rate']:.1f}%","","pos" if te['win_rate']>=60 else "neu"),
                        ("賺賠比",f"{te['pf']:.2f}","","pos" if te['pf']>=1 else "neg"),
                    ]):
                        c.markdown(kpi_card(l,v,s,cls), unsafe_allow_html=True)
                    cumulative_chart(test_b, "pnl_twd")

            if tr and te:
                robust = te["sharpe"] >= 0 and (tr["sharpe"] <= 0 or te["sharpe"] >= tr["sharpe"] * 0.5)
                if robust and te["sharpe"] >= 2:
                    st.markdown(f'<div class="verdict-ok">✅ 測試期 Sharpe {te["sharpe"]:.2f} 穩健（訓練期 {tr["sharpe"]:.2f}），策略跨期間表現一致</div>', unsafe_allow_html=True)
                elif robust:
                    st.markdown(f'<div class="verdict-warn">⚠️ 測試期 Sharpe {te["sharpe"]:.2f} 為正但偏低（訓練期 {tr["sharpe"]:.2f}），建議持續監控</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="verdict-bad">❌ 測試期 Sharpe {te["sharpe"]:.2f} 顯著低於訓練期 {tr["sharpe"]:.2f}，有過擬合風險</div>', unsafe_allow_html=True)

        # Monthly table
        if not trades_b.empty:
            sec("月份穩定性（最佳組合，2024-2026）")
            mth_b = monthly_table(trades_b, "pnl_twd")
            if not mth_b.empty:
                mth_b["損益(元)"] = mth_b["損益(元)"].apply(fmt_pnl)
                st.dataframe(apply_style(mth_b, wr_col="勝率", pf_col="賺賠比", pnl_cols=["損益(元)"]),
                             use_container_width=True, hide_index=True)

        # Overfitting
        sec("過擬合風險評估")
        col_ov1, col_ov2 = st.columns(2)
        with col_ov1:
            st.markdown("**各進場時間 最佳 Sharpe 分布**")
            bpt = (rdf_b.loc[rdf_b.groupby("trigger")["sharpe"].idxmax()]
                   [["trigger","sharpe"]].sort_values("trigger").set_index("trigger"))
            st.bar_chart(bpt, height=220)
            top10_med = rdf_b.nlargest(10,"sharpe")["sharpe"].median()
            top1_val  = float(best_b["sharpe"])
            st.markdown(f"**Top-1**: {top1_val:.2f} ｜ **Top-10 中位**: {top10_med:.2f}")
            if top1_val > top10_med * 1.8:
                st.warning("Top-1 遠高於 Top-10 中位，存在精選效應風險")
            else:
                st.success("Top-1 與 Top-10 中位接近，整體參數穩健")
        with col_ov2:
            st.markdown(f"**{best_b['trigger']} × stop={int(best_b['stop'])} × 各目標點 Sharpe**")
            trig_sub = (rdf_b[(rdf_b["trigger"]==best_b["trigger"]) & (rdf_b["stop"]==best_b["stop"])]
                        .groupby("target")["sharpe"].max().reset_index().set_index("target"))
            st.bar_chart(trig_sub, height=220)
            pos_tgt = (trig_sub["sharpe"] > 0).sum()
            total_tgt = len(trig_sub)
            if pos_tgt / total_tgt >= 0.7:
                st.success(f"止盈目標 {pos_tgt}/{total_tgt} 組 Sharpe > 0 ── 目標參數穩健")
            elif pos_tgt / total_tgt >= 0.5:
                st.warning(f"止盈目標 {pos_tgt}/{total_tgt} 組 Sharpe > 0 ── 部分區間不穩定")
            else:
                st.error(f"止盈目標 {pos_tgt}/{total_tgt} 組 Sharpe > 0 ── 參數敏感")

        # Filter + table
        sec("篩選器")
        cf1,cf2,cf3,cf4 = st.columns(4)
        trigs_b = ["全部"] + sorted(rdf_b["trigger"].unique().tolist())
        sel_tb  = cf1.selectbox("觸發時間", trigs_b, key="b_trig")
        sel_stp = cf2.selectbox("停損", ["全部","無停損(0)","有停損"], key="b_stp")
        min_sb  = cf3.slider("最低 Sharpe", -10.0, 15.0, 0.0, 0.5, key="b_shr")
        min_tp  = cf4.slider("最低 TP 率 (%)", 0, 100, 0, 5, key="b_tp")

        fdf_b = rdf_b.copy()
        if sel_tb != "全部":       fdf_b = fdf_b[fdf_b["trigger"] == sel_tb]
        if sel_stp == "無停損(0)": fdf_b = fdf_b[fdf_b["stop"] == 0]
        elif sel_stp == "有停損":  fdf_b = fdf_b[fdf_b["stop"] > 0]
        fdf_b = fdf_b[(fdf_b["sharpe"] >= min_sb) & (fdf_b["tp_rate"] >= min_tp)]

        COLS_B = ["trigger","target","stop","time_limit","trades","tp_rate","sl_rate",
                  "timeout_rate","win_rate","total_pnl","avg_pnl","profit_factor","expect_val","sharpe"]
        COLS_B = [c for c in COLS_B if c in fdf_b.columns]

        sec(f"全參數掃描（Sharpe 排序）— {len(fdf_b):,} 組符合")
        disp_b = fdf_b.sort_values("sharpe", ascending=False).head(30)[COLS_B].copy()
        for c in ["total_pnl","avg_pnl","expect_val"]:
            if c in disp_b.columns: disp_b[c] = disp_b[c].apply(fmt_pnl)
        for c in ["tp_rate","sl_rate","timeout_rate","win_rate"]:
            if c in disp_b.columns: disp_b[c] = disp_b[c].apply(fmt_pct)
        disp_b.columns = [c.replace("_"," ").title() for c in disp_b.columns]
        st.dataframe(apply_style(disp_b, sharpe_col="Sharpe", wr_col="Win Rate",
                                 pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl","Expect Val"]),
                     use_container_width=True, hide_index=True, height=480)

        sec("各進場時間最佳組合")
        bpt2 = rdf_b.loc[rdf_b.groupby("trigger")["sharpe"].idxmax()][
            [c for c in COLS_B if c in rdf_b.columns]].copy()
        for c in ["total_pnl","avg_pnl","expect_val"]:
            if c in bpt2.columns: bpt2[c] = bpt2[c].apply(fmt_pnl)
        for c in ["tp_rate","sl_rate","timeout_rate","win_rate"]:
            if c in bpt2.columns: bpt2[c] = bpt2[c].apply(fmt_pct)
        bpt2 = bpt2.sort_values("trigger")
        bpt2.columns = [c.replace("_"," ").title() for c in bpt2.columns]
        st.dataframe(apply_style(bpt2, sharpe_col="Sharpe", wr_col="Win Rate",
                                 pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl","Expect Val"]),
                     use_container_width=True, hide_index=True)

        sec("停損對比")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**{best_b['trigger']} × 各停損（target={int(best_b['target'])}）**")
            sub8 = (rdf_b[(rdf_b["trigger"]==best_b["trigger"]) & (rdf_b["target"]==best_b["target"])]
                    .sort_values(["stop","sharpe"], ascending=[True,False])
                    .drop_duplicates(subset=["stop"])[[c for c in COLS_B if c in rdf_b.columns]].copy())
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

        if not trades_b.empty:
            with st.expander("逐筆交易紀錄（最佳組合，按日期排序）"):
                dt_b = date_sorted_table(trades_b, "pnl_twd", ["result","pnl_pts"])
                dt_b.columns = [c.replace("_"," ").title() for c in dt_b.columns]
                st.dataframe(dt_b, use_container_width=True, hide_index=True)

        st.caption("⚠️ 夜盤方向濾網為合法前置資訊（前日 15:00～當日 05:00），無 Look-Ahead Bias")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — 策略B 出場優化
# ════════════════════════════════════════════════════════════════════════════
with tab_b_exit:
    sec("出場方式說明")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown("""
        **🎯 Full TP（現有）**
        - 全部倉位在 target 點停利
        - 未到停利點則 stop 或 timeout 出場
        - 優點：高 TP 率，低波動
        - 缺點：無法捕捉超額移動
        """)
    with col_e2:
        st.markdown("""
        **✂️ Partial TP + Trailing**
        - 50% 在 partial_at 點出場，止損移 BE
        - 剩餘 50% 等 full_target 或 timeout
        - 優點：鎖定一部分利潤，讓另一半跑
        - 缺點：若回到 BE 則只拿到一半
        """)
    with col_e3:
        st.markdown("""
        **📈 Pure Trailing Stop**
        - 無固定停利，跟隨最高點回落 trail_pts
        - 最高點不斷更新，止損也跟著拉高
        - 優點：捕捉大行情
        - 缺點：開盤衝量後若整理，提早出場
        """)

    rdf_e, trades_e = load_exit()

    if rdf_e.empty:
        st.info("⏳ 出場優化回測進行中，結果出來後自動更新（約 5 分鐘）")
        st.markdown("""
        **回測參數範圍：**
        - Full TP targets: 5, 8, 10, 12, 15, 20, 25, 30 pts
        - Partial TP: (partial_at, full_target) 共 10 組組合
        - Trailing stop: 3, 5, 8, 10, 12, 15, 20 pts
        - 固定進場：08:46 夜盤偏多，止損固定 5 pts
        """)
    else:
        # KPI summary
        best_by_mode = rdf_e.loc[rdf_e.groupby("mode")["sharpe"].idxmax()]

        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], best_by_mode.iterrows()):
            mode_name = {"full_tp": "Full TP", "partial_tp": "Partial TP", "trailing": "Trailing Stop"}.get(row["mode"], row["mode"])
            col.markdown(kpi_card(
                f"最佳 {mode_name}",
                f"Sharpe {row['sharpe']:.2f}",
                f"{row['label']}  EV={row['ev_twd']:+.1f}元",
                sharpe_cls(row["sharpe"])
            ), unsafe_allow_html=True)

        # Comparison table
        sec("三種出場方式全比較（Sharpe 排序）")
        disp_cols = ["mode","label","trades","win_rate","ev_twd","total_pnl","pf","sharpe"]
        disp_cols = [c for c in disp_cols if c in rdf_e.columns]
        disp_e = rdf_e.sort_values("sharpe", ascending=False)[disp_cols].copy()
        disp_e["total_pnl"] = disp_e["total_pnl"].apply(lambda x: f"{int(float(x)):+,}")
        disp_e["ev_twd"]    = disp_e["ev_twd"].apply(lambda x: f"{float(x):+.1f}")
        disp_e["win_rate"]  = disp_e["win_rate"].apply(fmt_pct)
        disp_e.columns      = [c.replace("_"," ").title() for c in disp_e.columns]
        st.dataframe(apply_style(disp_e, sharpe_col="Sharpe", pf_col="Pf",
                                 pnl_cols=["Total Pnl","Ev Twd"]),
                     use_container_width=True, hide_index=True, height=480)

        # Best per mode bar chart
        sec("各模式最佳 Sharpe 比較")
        bar_data = best_by_mode.set_index("mode")[["sharpe"]].rename(index={
            "full_tp": "Full TP", "partial_tp": "Partial TP", "trailing": "Trailing"})
        st.bar_chart(bar_data, height=200)

        # EV comparison
        sec("各模式最佳期望值（元/筆）比較")
        ev_data = best_by_mode.set_index("mode")[["ev_twd"]].rename(index={
            "full_tp": "Full TP", "partial_tp": "Partial TP", "trailing": "Trailing"})
        st.bar_chart(ev_data, height=200)

        # Monthly tables for best of each mode
        if not trades_e.empty:
            sec("月份穩定性（各模式最佳組合）")
            tabs_modes = st.tabs(["Full TP", "Partial TP", "Trailing Stop"])
            for tab_m, mode in zip(tabs_modes, ["full_tp", "partial_tp", "trailing"]):
                with tab_m:
                    best_lbl = rdf_e[rdf_e["mode"] == mode].sort_values("sharpe", ascending=False).iloc[0]["label"]
                    t_mode   = trades_e[trades_e["label"] == best_lbl].copy()
                    st.markdown(f"**{best_lbl}**")
                    col_chart, col_mth = st.columns([1, 1])
                    with col_chart:
                        cumulative_chart(t_mode, "pnl_pts")
                    with col_mth:
                        mth_e = monthly_table(t_mode.assign(pnl_twd=t_mode["pnl_pts"] * 10), "pnl_twd")
                        if not mth_e.empty:
                            mth_e["損益(元)"] = mth_e["損益(元)"].apply(fmt_pnl)
                            st.dataframe(apply_style(mth_e, wr_col="勝率", pf_col="賺賠比",
                                                     pnl_cols=["損益(元)"]),
                                         use_container_width=True, hide_index=True)

        # Conclusion
        sec("結論與建議")
        if not rdf_e.empty:
            best_overall = rdf_e.loc[rdf_e["sharpe"].idxmax()]
            mode_zh = {"full_tp": "Full TP（全倉停利）", "partial_tp": "Partial TP（部分停利）",
                       "trailing": "Trailing Stop（移動停利）"}.get(best_overall["mode"], best_overall["mode"])
            st.markdown(f"""
            <div class="verdict-ok">
            ✅ 最佳出場方式：<strong>{mode_zh}</strong>（{best_overall['label']}）<br>
            Sharpe {best_overall['sharpe']:.2f}｜期望值 {best_overall['ev_twd']:+.1f} 元/筆｜
            勝率 {best_overall['win_rate']:.1f}%｜賺賠比 {best_overall['pf']:.2f}
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — 策略A
# ════════════════════════════════════════════════════════════════════════════
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
            ("Sharpe 值",    f"{best_a['sharpe']:.2f}",     f"{best_a['trigger']}", "pos"),
            ("總損益",        f"{tp_a:+,} 元",                "最高 Sharpe 組合", "pos" if tp_a>0 else "neg"),
            ("勝率",          fmt_pct(best_a["win_rate"]),    "最高 Sharpe 組合", ""),
            ("賺賠比",        f"{pf_a:.2f}",                  "profit factor", "pos" if float(pf_a)>=1 else "neg"),
            ("期望值(元/筆)", f"{int(float(ev_a)):+,}",       "每筆平均損益", "pos" if float(ev_a)>0 else "neg"),
            ("正Sharpe 組數", f"{(rdf_a['sharpe']>0).sum()}", f"共 {len(rdf_a)} 組", ""),
        ]
        for col,(lbl,val,sub,cls) in zip([c1,c2,c3,c4,c5,c6], cards_a):
            col.markdown(kpi_card(lbl,val,sub,cls), unsafe_allow_html=True)

        if not trades_a.empty:
            sec("最佳組合累積損益曲線")
            cumulative_chart(trades_a, "pnl_twd")

        if not trades_a.empty:
            sec("月份穩定性（最佳組合）")
            mth_a = monthly_table(trades_a, "pnl_twd")
            if not mth_a.empty:
                mth_a["損益(元)"] = mth_a["損益(元)"].apply(fmt_pnl)
                st.dataframe(apply_style(mth_a, wr_col="勝率", pf_col="賺賠比", pnl_cols=["損益(元)"]),
                             use_container_width=True, hide_index=True)

        sec("篩選器")
        cf1,cf2,cf3 = st.columns(3)
        triggers_a = ["全部"] + sorted(rdf_a["trigger"].unique().tolist())
        sel_trig_a = cf1.selectbox("觸發時間", triggers_a, key="a_trig")
        min_shr_a  = cf2.slider("最低 Sharpe", -5.0, 15.0, 0.0, 0.5, key="a_shr")
        min_wr_a   = cf3.slider("最低勝率 (%)", 0, 100, 0, 5, key="a_wr")

        fdf_a = rdf_a.copy()
        if sel_trig_a != "全部": fdf_a = fdf_a[fdf_a["trigger"] == sel_trig_a]
        fdf_a = fdf_a[(fdf_a["sharpe"] >= min_shr_a) & (fdf_a["win_rate"] >= min_wr_a)]

        COLS_A = ["trigger","offset","target","stop","time_limit","trades","win_rate",
                  "total_pnl","avg_pnl","profit_factor","expect_val","sharpe"]
        COLS_A = [c for c in COLS_A if c in fdf_a.columns]

        sec(f"Top 30（Sharpe）— {len(fdf_a):,} 組符合")
        disp_a = fdf_a.sort_values("sharpe", ascending=False).head(30)[COLS_A].copy()
        for c in ["total_pnl","avg_pnl","expect_val"]:
            if c in disp_a.columns: disp_a[c] = disp_a[c].apply(fmt_pnl)
        if "win_rate" in disp_a.columns: disp_a["win_rate"] = disp_a["win_rate"].apply(fmt_pct)
        disp_a.columns = [c.replace("_"," ").title() for c in disp_a.columns]
        st.dataframe(apply_style(disp_a, sharpe_col="Sharpe", wr_col="Win Rate",
                                 pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl","Expect Val"]),
                     use_container_width=True, hide_index=True, height=480)

        sec("各觸發時間最佳組合")
        bpa = rdf_a.loc[rdf_a.groupby("trigger")["sharpe"].idxmax()][COLS_A].copy()
        for c in ["total_pnl","avg_pnl","expect_val"]:
            if c in bpa.columns: bpa[c] = bpa[c].apply(fmt_pnl)
        if "win_rate" in bpa.columns: bpa["win_rate"] = bpa["win_rate"].apply(fmt_pct)
        bpa.columns = [c.replace("_"," ").title() for c in bpa.columns]
        st.dataframe(apply_style(bpa, sharpe_col="Sharpe", wr_col="Win Rate",
                                 pf_col="Profit Factor", pnl_cols=["Total Pnl","Avg Pnl","Expect Val"]),
                     use_container_width=True, hide_index=True)

        if not trades_a.empty:
            with st.expander("逐筆交易紀錄（最佳組合，按日期排序）"):
                dt_a = date_sorted_table(trades_a, "pnl_twd", ["direction","pnl_pts"])
                dt_a.columns = [c.replace("_"," ").title() for c in dt_a.columns]
                st.dataframe(dt_a, use_container_width=True, hide_index=True)

        st.caption("⚠️ 參數掃描結果（3,920 組），請注意過擬合風險")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — 策略C/D 股票策略（合併）
# ════════════════════════════════════════════════════════════════════════════
with tab_cd:
    st.markdown("""
    <div style="background:#1c2128;border:1px solid #f85149;border-radius:8px;padding:12px 16px;margin:8px 0;">
    ❌ <strong>策略C/D 均不建議使用</strong><br>
    策略C：前日強勢股 Gap-Fade 效應，全面虧損。<br>
    策略D：台積電手續費陷阱，股價<1,000元時連TP也虧損。
    </div>
    """, unsafe_allow_html=True)

    tab_c_inner, tab_d_inner = st.tabs(["策略C：前日強勢大型股", "策略D：台積電單股"])

    # ── 策略C ──
    with tab_c_inner:
        sec("策略邏輯")
        st.markdown("""
        - **選股**：前一日漲幅前 30 名（市值 > 500 億）
        - **MXF 方向濾網**：08:46–08:59 偏多才交易
        - **進場**：09:01 開盤第一根 K 棒（漲停略過）
        - **停利**：N tick，時間窗口內達到即觸發
        - **手續費**：1.6折 + 當沖稅減半
        """)

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
            c3.markdown(kpi_card("Tick 目標", f"{ticks[0]}~{ticks[-1]}" if ticks else "—", "", ""), unsafe_allow_html=True)
            c4.markdown(kpi_card("資料品質", "1-min K 棒" if is_1min else "日頻", "", "pos" if is_1min else "neu"), unsafe_allow_html=True)

            summary_rows = []
            for n in ticks:
                for tw in (windows if is_1min else [None]):
                    mask = rdf_c["tick"] == n
                    if is_1min and tw: mask &= rdf_c["time_window"] == tw
                    t = rdf_c[mask]
                    if t.empty: continue
                    wins  = (t["pnl"] > 0).sum()
                    loss  = (t["pnl"] < 0).sum()
                    tp_n  = (t["result"] == "TP").sum()
                    total = len(t)
                    std   = t["pnl"].std()
                    sharpe = t["pnl"].mean() / std * np.sqrt(252) if std > 0 else 0
                    wa = t[t["pnl"]>0]["pnl"].mean() if wins else 0
                    la = t[t["pnl"]<0]["pnl"].mean() if loss else 0
                    pf = abs(wa*wins/(la*loss)) if loss>0 and la!=0 else 99.0
                    row = {"tick":n,"筆數":total,"TP率":fmt_pct(tp_n/total*100),
                           "勝率":fmt_pct(wins/total*100),"賺賠比":round(pf,2),
                           "期望值(元)":round(t["pnl"].mean(),0),"總損益(元)":int(t["pnl"].sum()),
                           "Sharpe":round(sharpe,2)}
                    if is_1min: row["時間窗口(min)"] = tw
                    summary_rows.append(row)

            if summary_rows:
                sdf = pd.DataFrame(summary_rows)
                sdf_d = sdf.copy()
                sdf_d["總損益(元)"] = sdf_d["總損益(元)"].apply(fmt_pnl)
                st.dataframe(apply_style(sdf_d, sharpe_col="Sharpe", wr_col="勝率",
                                         pf_col="賺賠比", pnl_cols=["總損益(元)"]),
                             use_container_width=True, hide_index=True)

            col_ga, col_gb = st.columns(2)
            with col_ga:
                st.markdown("""
                **為什麼全面虧損？**
                - 前日強勢股已反映漲幅，隔日開盤溢價 gap
                - 開盤後主力出貨，散戶追高被套
                - TIMEOUT 平均虧損遠大於 TP 獲利
                """)
            with col_gb:
                st.markdown("""
                **改進方向**
                1. 做空前日強勢股（需現股當沖資格）
                2. 做多前日弱勢股（均值回歸）
                3. 09:10~09:20 等支撐確認後進場
                """)

    # ── 策略D ──
    with tab_d_inner:
        sec("策略邏輯")
        st.markdown("""
        - **選股**：只做台積電（2330）
        - **進場條件**：MXF 08:46–08:59 偏多才進場（09:01 開盤第一根）
        - **資料期間**：2024-01-02 ~ 2026-05-22（576 個交易日）
        """)

        st.markdown("""
        <div style="background:#1c2128;border:1px solid #f0883e;border-radius:8px;padding:12px 16px;margin:8px 0;">
        ⚠️ <strong>手續費陷阱：</strong>台積電 800 元時，1 tick = NT$1,000 但手續費 ≈ NT$1,565 → TP 也虧損。<br>
        只有台積電 <strong>>1,000 元</strong>（tick=5元）才有足夠利潤空間。
        </div>
        """, unsafe_allow_html=True)

        rdf_t = load_tsmc()
        if not rdf_t.empty:
            ticks_t   = sorted(rdf_t["tick"].unique().tolist()) if "tick" in rdf_t.columns else []
            windows_t = sorted(rdf_t["time_window"].unique().tolist()) if "time_window" in rdf_t.columns else []
            tdays_t   = rdf_t["date"].nunique() if "date" in rdf_t.columns else 0

            summary_t = []
            for n in ticks_t:
                for tw in windows_t:
                    t = rdf_t[(rdf_t["tick"]==n) & (rdf_t["time_window"]==tw)]
                    if t.empty: continue
                    wins = (t["pnl"]>0).sum(); loss=(t["pnl"]<0).sum(); total=len(t)
                    tp_n = (t["result"]=="TP").sum()
                    std  = t["pnl"].std()
                    sharpe = t["pnl"].mean()/std*np.sqrt(252) if std>0 else 0
                    wa = t[t["pnl"]>0]["pnl"].mean() if wins else 0
                    la = t[t["pnl"]<0]["pnl"].mean() if loss else 0
                    pf = abs(wa*wins/(la*loss)) if loss>0 and la!=0 else 99.0
                    summary_t.append({"tick":n,"時間窗口(min)":tw,"交易日":total,
                                      "TP率":f"{tp_n/total*100:.1f}%","勝率":f"{wins/total*100:.1f}%",
                                      "賺賠比":round(pf,2),"Sharpe":round(sharpe,2),
                                      "期望值(元)":round(t["pnl"].mean(),0),"總損益(元)":int(t["pnl"].sum())})

            c1,c2,c3 = st.columns(3)
            c1.markdown(kpi_card("回測交易日", str(tdays_t), "MXF 偏多", ""), unsafe_allow_html=True)
            best_t = max(summary_t, key=lambda r: float(r["Sharpe"])) if summary_t else {}
            c2.markdown(kpi_card("最佳 Sharpe",
                                 f"{best_t.get('Sharpe','—'):.2f}" if best_t else "—",
                                 f"{best_t.get('tick','?')} tick × {best_t.get('時間窗口(min)','?')} min" if best_t else "",
                                 "pos" if best_t and float(best_t.get("Sharpe","-99"))>0 else "neg"),
                        unsafe_allow_html=True)
            c3.markdown(kpi_card("價格範圍","574 ~ 2,340 元","tick=1(≤1000) / tick=5(>1000)","neu"),
                        unsafe_allow_html=True)

            if summary_t:
                sdf_t = pd.DataFrame(summary_t)
                sdf_t["總損益(元)"] = sdf_t["總損益(元)"].apply(fmt_pnl)
                st.dataframe(apply_style(sdf_t, sharpe_col="Sharpe", wr_col="勝率",
                                         pf_col="賺賠比", pnl_cols=["總損益(元)"]),
                             use_container_width=True, hide_index=True)
