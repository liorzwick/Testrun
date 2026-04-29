import pandas as pd
import yfinance as yf
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
import time
from tqdm import tqdm
import urllib.request

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
@dataclass
class BacktestConfig:
    start_year: int = 2015
    end_year: int = 2025
    benchmark: str = "SPY"
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.005      
    max_alloc_pct: float = 0.12        # עד 12% מהתיק למניה
    max_positions: int = 10            # <<< מקסימום 10 פוזיציות פתוחות (אכיפה קשיחה)
    max_portfolio_heat: float = 0.04   # מקסימום סיכון של 4% על כל התיק יחד
    cooldown_days: int = 15
    slippage_bps: float = 12
    commission_bps: float = 2
    
    breakout_volume_ratio: float = 1.1 
    min_dollar_vol_50: float = 25_000_000
    min_price: float = 15.0
    
    # ניהול טרייד והגבלת סטופ
    min_risk_pct: float = 0.01         
    max_risk_pct: float = 0.06         
    max_hold_bars: int = 250           
    time_stop_bars: int = 35           
    min_profit_after_time_stop: float = 0.01 
    
    # פרמטרים גיאומטריים של משולש עולה (VCP)
    min_prior_uptrend: float = 0.08    
    max_base_depth: float = 0.35       
    max_tightness_depth: float = 0.08  
    min_breakout_close_strength: float = 0.30
    min_rs_65: float = 0.02            # העדפה למניות חזקות מהמדד
    max_dist_from_52w_high: float = 0.15
    
    min_cup_depth: float = 0.04
    max_cup_depth: float = 0.35
    max_handle_depth: float = 0.10
    min_handle_days: int = 3
    max_handle_days: int = 20
    min_cup_days: int = 15
    max_cup_days: int = 200
    max_pivot_extension: float = 0.04  
    max_entry_extension: float = 0.03  
    max_gap_above_pivot: float = 0.02
    
    early_exit_bars: int = 10          
    early_exit_min_progress: float = -0.02 
    min_tight_closes_in_handle: int = 0
    
    use_point_in_time_universe: bool = False
    raw_price_mode: bool = False
    allow_same_day_cash_reuse: bool = False
    universe_file: str | None = None
    output_prefix: str = "v13_strict_portfolio"

# ==========================================
# 1. Data & Indicators
# ==========================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    
    df = df[~df.index.duplicated(keep='first')]

    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_150"] = df["Close"].rolling(150).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Vol_50"] = df["Volume"].rolling(50).mean()
    df["DollarVol_50"] = df["Close"].rolling(50).mean() * df["Volume"].rolling(50).mean()
    df["Prev_Close"] = df["Close"].shift(1)
    df["ROC_65"] = df["Close"].pct_change(65)
    df["High_252"] = df["High"].rolling(252).max()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Prev_Close"]).abs(), (df["Low"] - df["Prev_Close"]).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    return df

def get_data(ticker: str, start_fetch: str, end_fetch: str, cfg: BacktestConfig) -> pd.DataFrame:
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start_fetch}_{end_fetch}_v13.pkl"
    if cache_file.exists(): return pd.read_pickle(cache_file)
    try:
        df = yf.download(ticker, start=start_fetch, end=end_fetch, auto_adjust=True, progress=False)
        if not df.empty:
            df = add_indicators(df)
            df.to_pickle(cache_file)
            return df
    except: pass
    return pd.DataFrame()

# ==========================================
# 3. Filters
# ==========================================
def market_filter_ok(spy_df: pd.DataFrame, current_date: pd.Timestamp) -> bool:
    x = spy_df[spy_df.index <= current_date]
    if len(x) < 220: return False
    last_row = x.iloc[-1]
    return float(last_row["Close"]) > float(last_row["SMA_200"])

def stock_filter_ok(today: pd.Series, cfg: BacktestConfig) -> bool:
    for col in ["SMA_200", "Vol_50", "High_252"]:
        if pd.isna(today[col]).any() if isinstance(today[col], pd.Series) else pd.isna(today[col]):
            return False

    close_p = float(today["Close"])
    if close_p < cfg.min_price or float(today["DollarVol_50"]) < cfg.min_dollar_vol_50: 
        return False

    dist_52w = (close_p / float(today["High_252"])) - 1.0
    return dist_52w >= -cfg.max_dist_from_52w_high

# ==========================================
# 4. Pattern Detection (Ascending Triangle)
# ==========================================
def get_ascending_triangle_signal(pattern_data: pd.DataFrame, cfg: BacktestConfig):
    recent = pattern_data.tail(150)
    if len(recent) < 60: return None

    highs = recent["High"].values
    lows = recent["Low"].values

    slice1 = highs[:-10]
    if len(slice1) == 0: return None
    peak1_pos = int(np.argmax(slice1))
    peak1_price = float(slice1[peak1_pos])

    if peak1_pos < 30: return None 

    post_peak1_lows = lows[peak1_pos+1 : -5]
    if len(post_peak1_lows) < 10: return None

    valley1_pos_rel = int(np.argmin(post_peak1_lows))
    valley1_price = float(post_peak1_lows[valley1_pos_rel])
    valley1_pos = peak1_pos + 1 + valley1_pos_rel

    base_depth = (peak1_price - valley1_price) / peak1_price
    if base_depth > cfg.max_base_depth: return None

    post_valley1_highs = highs[valley1_pos+1 : -2]
    if len(post_valley1_highs) < 5: return None

    peak2_pos_rel = int(np.argmax(post_valley1_highs))
    peak2_price = float(post_valley1_highs[peak2_pos_rel])
    peak2_pos = valley1_pos + 1 + peak2_pos_rel

    if peak2_price < peak1_price * 0.96 or peak2_price > peak1_price * 1.02:
        return None

    post_peak2_lows = lows[peak2_pos+1 :]
    if len(post_peak2_lows) < 3: return None

    valley2_price = float(np.min(post_peak2_lows))
    
    if valley2_price <= valley1_price * 1.015: 
        return None

    tightness = (peak2_price - valley2_price) / peak2_price
    if tightness > cfg.max_tightness_depth: 
        return None

    pivot = max(peak1_price, peak2_price)

    return {"pivot_price": pivot, "tight_low": valley2_price}

# ==========================================
# 5. Patient Trade Simulation
# ==========================================
def simulate_trade(df: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float, initial_stop: float, cfg: BacktestConfig):
    future = df[df.index >= entry_date].head(cfg.max_hold_bars)
    if future.empty: return None
    
    stop_price = float(initial_stop)
    if stop_price >= entry_price:
        stop_price = entry_price * 0.985 
        
    highest_seen = float(entry_price)

    for i, row in enumerate(future.itertuples()):
        day_low = float(row.Low)
        day_high = float(row.High)
        day_close = float(row.Close)

        highest_seen = max(highest_seen, day_high)
        profit = (highest_seen / entry_price) - 1

        if profit >= 0.08: stop_price = max(stop_price, entry_price * 1.005) 
        if profit >= 0.15: stop_price = max(stop_price, highest_seen * 0.90) 
        if profit >= 0.30: stop_price = max(stop_price, highest_seen * 0.85) 

        if day_low <= stop_price:
            exit_p = min(float(row.Open), stop_price) * (1 - cfg.slippage_bps/10000)
            return {"Exit_Date": row.Index, "Exit_Price": exit_p, "Exit_Reason": "StopHit", "Hold": i+1}

        if i >= cfg.time_stop_bars and (day_close/entry_price-1) < cfg.min_profit_after_time_stop:
            return {"Exit_Date": row.Index, "Exit_Price": day_close, "Exit_Reason": "TimeExit", "Hold": i+1}

        if i == cfg.early_exit_bars and (day_close/entry_price-1) < cfg.early_exit_min_progress:
            return {"Exit_Date": row.Index, "Exit_Price": day_close, "Exit_Reason": "EarlyFail", "Hold": i+1}

    return {"Exit_Date": future.index[-1], "Exit_Price": float(future.iloc[-1].Close), "Exit_Reason": "MaxHold", "Hold": len(future)}

# ==========================================
# 6. Candidate Generation
# ==========================================
def generate_candidates(tickers, data_cache, spy_df, cfg):
    candidates = []
    # סורק את כל הרשימה שהועברה לו
    for ticker in tqdm(tickers, desc="Scanning S&P 500"):
        df = data_cache.get(ticker, pd.DataFrame())
        if df.empty or len(df) < 250: continue
        for i in range(250, len(df)):
            past = df.iloc[:i]
            today = past.iloc[-1]
            if today.name.year < cfg.start_year or not market_filter_ok(spy_df, today.name): continue
            if not stock_filter_ok(today, cfg): continue
            
            spy_past = spy_df[spy_df.index <= today.name]
            if not spy_past.empty:
                spy_rs = float(spy_past.iloc[-1]["ROC_65"])
                stock_rs = float(today["ROC_65"])
                if (stock_rs - spy_rs) < cfg.min_rs_65:
                    continue
            
            pattern = get_ascending_triangle_signal(past, cfg)
            if pattern:
                curr_close = float(today.Close)
                prev_close = float(past.iloc[-2].Close)
                pivot = pattern['pivot_price']
                
                if curr_close > pivot and prev_close <= pivot:
                    if curr_close <= pivot * (1+cfg.max_pivot_extension):
                        if i+1 >= len(df): continue
                        
                        entry_p = float(df.iloc[i+1].Open) * (1+cfg.slippage_bps/10000)
                        
                        atr_val = float(today["ATR_14"])
                        tight_low = float(pattern["tight_low"])
                        calculated_stop = tight_low - (0.5 * atr_val)
                        max_allowed_stop = entry_p * (1 - cfg.max_risk_pct)
                        initial_stop = max(calculated_stop, max_allowed_stop)
                        
                        sim = simulate_trade(df, df.index[i+1], entry_p, initial_stop, cfg)
                        
                        if sim:
                            candidates.append({
                                "Year": today.name.year, "Ticker": ticker, "Entry_Date": df.index[i+1], 
                                "Exit_Date": sim["Exit_Date"], "Entry_Price": entry_p, "Exit_Price": sim["Exit_Price"], 
                                "Pct": (sim["Exit_Price"]/entry_p-1)*100, "Reason": sim["Exit_Reason"],
                                "Stop_Price": initial_stop
                            })
    if not candidates: return pd.DataFrame()
    return pd.DataFrame(candidates).sort_values("Entry_Date").reset_index(drop=True)

# ==========================================
# 7. Portfolio Management (STRICT RULES)
# ==========================================
def get_close_on_or_before(df: pd.DataFrame, dt: pd.Timestamp, fallback: float) -> float:
    x = df[df.index <= dt]
    return float(x.iloc[-1]["Close"]) if not x.empty else fallback

def accept_trades_with_portfolio_rules(candidates: pd.DataFrame, data_cache: dict, cfg: BacktestConfig) -> pd.DataFrame:
    if candidates.empty: return pd.DataFrame()

    cash = cfg.initial_capital
    active = []
    accepted = []
    last_exit_by_ticker = {}

    cand_records = candidates.to_dict("records")
    for cand in cand_records:
        entry_date = pd.Timestamp(cand["Entry_Date"])
        ticker = str(cand["Ticker"])

        if ticker in last_exit_by_ticker and entry_date <= last_exit_by_ticker[ticker] + pd.Timedelta(days=cfg.cooldown_days):
            continue

        release, still_active = [], []
        for pos in active:
            exit_dt = pd.Timestamp(pos["Exit_Date"])
            closed = (exit_dt < entry_date or (cfg.allow_same_day_cash_reuse and exit_dt == entry_date))
            if closed:
                release.append(pos)
            else:
                still_active.append(pos)

        for pos in release:
            cash += pos["Shares"] * pos["Exit_Price"] - pos["Exit_Fee"]
        active = still_active

        # >>> אכיפת מקסימום פוזיציות ואיסור כפילויות <<<
        if any(pos["Ticker"] == ticker for pos in active): continue
        if len(active) >= cfg.max_positions: continue

        equity = cash + sum(
            get_close_on_or_before(data_cache[p["Ticker"]], entry_date, p["Entry_Price"]) * p["Shares"]
            for p in active
        )

        entry_price = float(cand["Entry_Price"])
        stop_price = float(cand["Stop_Price"])
        exit_price = float(cand["Exit_Price"])

        risk_per_share = max(entry_price - stop_price, 1e-9)
        max_risk_dollars_trade = equity * cfg.risk_per_trade
        current_heat = sum(p["Risk_Dollars"] for p in active)
        max_heat = equity * cfg.max_portfolio_heat
        remaining_heat = max(0.0, max_heat - current_heat)

        if remaining_heat <= 0: continue

        shares_by_risk = min(max_risk_dollars_trade, remaining_heat) / risk_per_share
        shares_by_alloc = (equity * cfg.max_alloc_pct) / entry_price
        
        # אכיפת מזומן חזקה (כדי שהתיק לא ייכנס למינוס)
        shares_by_cash = cash / (entry_price * (1 + cfg.commission_bps / 10000))

        shares = int(np.floor(min(shares_by_risk, shares_by_alloc, shares_by_cash)))
        if shares < 1: continue

        entry_fee = shares * entry_price * cfg.commission_bps / 10000
        total_cost = shares * entry_price + entry_fee
        
        if total_cost > cash: continue

        exit_fee = shares * exit_price * cfg.commission_bps / 10000
        risk_dollars = shares * risk_per_share
        cash -= total_cost

        t = cand.copy()
        t["Shares"] = shares
        t["Entry_Fee"] = round(entry_fee, 2)
        t["Exit_Fee"] = round(exit_fee, 2)
        t["Gross_Entry"] = round(shares * entry_price, 2)
        t["Gross_Exit"] = round(shares * exit_price, 2)
        t["Net_PnL"] = round((shares * (exit_price - entry_price)) - entry_fee - exit_fee, 2)
        t["Alloc_Pct"] = round(shares * entry_price / equity * 100, 2) if equity > 0 else 0.0
        t["Risk_Dollars"] = round(risk_dollars, 2)
        
        accepted.append(t)
        last_exit_by_ticker[ticker] = pd.Timestamp(t["Exit_Date"])
        active.append({
            "Ticker": ticker, "Entry_Date": t["Entry_Date"], "Exit_Date": t["Exit_Date"],
            "Entry_Price": entry_price, "Exit_Price": exit_price, "Shares": shares, "Exit_Fee": exit_fee,
            "Risk_Dollars": risk_dollars,
        })

    if not accepted: return pd.DataFrame()
    return pd.DataFrame(accepted).sort_values(["Entry_Date", "Exit_Date", "Ticker"]).reset_index(drop=True)

# ==========================================
# 8. Daily Equity Curve
# ==========================================
def build_daily_equity_curve(accepted_df: pd.DataFrame, data_cache: dict, benchmark_df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if accepted_df.empty:
        return pd.DataFrame(columns=["Date", "Cash", "Market_Value", "Equity", "Drawdown_Pct", "Open_Positions"])

    start_dt = pd.Timestamp(f"{cfg.start_year}-01-01")
    end_dt = pd.Timestamp(f"{cfg.end_year}-12-31")

    accepted_records = accepted_df.to_dict("records")
    trade_dates = [pd.Timestamp(r["Entry_Date"]) for r in accepted_records] + [pd.Timestamp(r["Exit_Date"]) for r in accepted_records]
    base_calendar = benchmark_df.index
    full_calendar = base_calendar.union(pd.DatetimeIndex(trade_dates)).drop_duplicates().sort_values()
    calendar = full_calendar[(full_calendar >= start_dt) & (full_calendar <= end_dt)]

    entries_by_date, exits_by_date = {}, {}
    for r in accepted_records:
        entries_by_date.setdefault(pd.Timestamp(r["Entry_Date"]), []).append(r)
        exits_by_date.setdefault(pd.Timestamp(r["Exit_Date"]), []).append(r)

    cash = cfg.initial_capital
    open_pos, rows = {}, []
    running_peak = cfg.initial_capital

    for dt in calendar:
        if cfg.allow_same_day_cash_reuse:
            for r in exits_by_date.get(dt, []):
                key = (r["Ticker"], pd.Timestamp(r["Entry_Date"]), pd.Timestamp(r["Exit_Date"]))
                if key in open_pos:
                    cash += float(r["Gross_Exit"]) - float(r["Exit_Fee"])
                    del open_pos[key]
            for r in entries_by_date.get(dt, []):
                key = (r["Ticker"], pd.Timestamp(r["Entry_Date"]), pd.Timestamp(r["Exit_Date"]))
                open_pos[key] = r
                cash -= float(r["Gross_Entry"]) + float(r["Entry_Fee"])
        else:
            for r in entries_by_date.get(dt, []):
                key = (r["Ticker"], pd.Timestamp(r["Entry_Date"]), pd.Timestamp(r["Exit_Date"]))
                open_pos[key] = r
                cash -= float(r["Gross_Entry"]) + float(r["Entry_Fee"])
            for r in exits_by_date.get(dt, []):
                key = (r["Ticker"], pd.Timestamp(r["Entry_Date"]), pd.Timestamp(r["Exit_Date"]))
                if key in open_pos:
                    cash += float(r["Gross_Exit"]) - float(r["Exit_Fee"])
                    del open_pos[key]

        market_value = sum(
            get_close_on_or_before(data_cache[pos["Ticker"]], dt, float(pos["Entry_Price"])) * float(pos["Shares"])
            for pos in open_pos.values()
        )
        equity = cash + market_value
        running_peak = max(running_peak, equity)
        dd = (equity / running_peak - 1.0) * 100 if running_peak > 0 else 0.0

        rows.append({
            "Date": dt, "Cash": round(cash, 2), "Market_Value": round(market_value, 2),
            "Equity": round(equity, 2), "Drawdown_Pct": round(dd, 2), "Positions": len(open_pos),
        })

    return pd.DataFrame(rows)

# ==========================================
# 9. Summaries
# ==========================================
def calc_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) == 0: return 0.0
    dd = (equity_curve / equity_curve.cummax()) - 1.0
    return round(dd.min() * 100, 2)

def summarize_trades(trades_df: pd.DataFrame, equity_df: pd.DataFrame | None = None) -> dict:
    empty = {
        "Trades": 0, "Wins": 0, "Losses": 0, "Win_Rate_Pct": 0.0, 
        "Avg_Trade_Pct": 0.0, "Avg_Win_Pct": 0.0, "Avg_Loss_Pct": 0.0,
        "Total_Return_Pct": 0.0, "Max_Drawdown_Pct": 0.0, "Net_PnL": 0.0
    }
    if trades_df.empty: return empty

    wins = trades_df[trades_df["Pct"] > 0]
    losses = trades_df[trades_df["Pct"] < 0]

    total_return, max_dd = 0.0, 0.0
    if equity_df is not None and not equity_df.empty:
        total_return = round((equity_df["Equity"].iloc[-1] / equity_df["Equity"].iloc[0] - 1.0) * 100, 2)
        max_dd = calc_drawdown(equity_df["Equity"])

    return {
        "Trades": len(trades_df), "Wins": len(wins), "Losses": len(losses),
        "Win_Rate_Pct": round(len(wins) / len(trades_df) * 100, 2) if len(trades_df) > 0 else 0.0,
        "Avg_Trade_Pct": round(trades_df["Pct"].mean(), 2),
        "Avg_Win_Pct": round(wins["Pct"].mean(), 2) if len(wins) > 0 else 0.0,
        "Avg_Loss_Pct": round(losses["Pct"].mean(), 2) if len(losses) > 0 else 0.0,
        "Total_Return_Pct": total_return, "Max_Drawdown_Pct": max_dd,
        "Net_PnL": round(trades_df["Net_PnL"].sum(), 2) if "Net_PnL" in trades_df.columns else 0.0,
    }

def yearly_summary(accepted_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    if accepted_df.empty: return pd.DataFrame()
    tmp = accepted_df.copy()
    tmp["Entry_Date"] = pd.to_datetime(tmp["Entry_Date"])
    tmp["Year"] = tmp["Entry_Date"].dt.year
    rows = []
    for year, g in tmp.groupby("Year"):
        eq = equity_df[equity_df["Date"].dt.year == year] if not equity_df.empty else pd.DataFrame()
        s = summarize_trades(g, eq)
        s["Year"] = year
        rows.append(s)
    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)

def monthly_summary(accepted_df: pd.DataFrame) -> pd.DataFrame:
    if accepted_df.empty: return pd.DataFrame()
    tmp = accepted_df.copy()
    tmp["Entry_Date"] = pd.to_datetime(tmp["Entry_Date"])
    tmp["Month"] = tmp["Entry_Date"].dt.to_period("M").astype(str)
    return (
        tmp.groupby("Month")
        .agg(
            Trades=("Ticker", "count"),
            Win_Rate_Pct=("Pct", lambda s: round(s.gt(0).sum() / len(s) * 100, 2) if len(s) > 0 else 0),
            Avg_Trade_Pct=("Pct", lambda s: round(s.mean(), 2)),
            Net_PnL=("Net_PnL", "sum"),
        )
        .reset_index()
    )

# ==========================================
# 10. Orchestrator
# ==========================================
def run_backtest_engine(tickers, cfg):
    spy = get_data(cfg.benchmark, "2014-01-01", "2026-03-01", cfg)
    data_cache = {cfg.benchmark: spy}
    
    # >>> סריקה מלאה של כל הטיקרים ללא חיתוך <<<
    for t in tqdm(tickers, desc="Loading Data"):
        data_cache[t] = get_data(t, "2014-01-01", "2026-03-01", cfg)
        
    cands = generate_candidates(tickers, data_cache, spy, cfg)
    acc = accept_trades_with_portfolio_rules(cands, data_cache, cfg)
    eq = build_daily_equity_curve(acc, data_cache, spy, cfg)
    
    yearly_df = yearly_summary(acc, eq)
    monthly_df = monthly_summary(acc)
    overall = summarize_trades(acc, eq)
    
    return cands, acc, eq, yearly_df, monthly_df, overall

# ==========================================
# 11. Output Helpers
# ==========================================
def save_outputs(candidates_df, accepted_df, equity_df, yearly_df, monthly_df, overall, cfg: BacktestConfig):
    out_dir = Path("output") / cfg.output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_df.to_csv(out_dir / "candidate_signals.csv", index=False, encoding="utf-8-sig")
    accepted_df.to_csv(out_dir / "accepted_trades.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(out_dir / "yearly_summary.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(out_dir / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([overall]).to_csv(out_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([asdict(cfg)]).to_csv(out_dir / "config.csv", index=False, encoding="utf-8-sig")
    print(f"\nFiles saved -> {out_dir}/")

def print_final_report(overall: dict, yearly_df: pd.DataFrame):
    if not overall or overall.get('Trades', 0) == 0:
        print("No trades executed.")
        return
    print("\n" + "=" * 80)
    print("STRICT PORTFOLIO VCP BACKTEST REPORT (v13)")
    print("=" * 80)
    for _, r in yearly_df.iterrows():
        print(f" {int(r['Year'])}: trades={int(r['Trades']):3d} | WR={r['Win_Rate_Pct']:5.1f}% | avgTrade={r['Avg_Trade_Pct']:+5.2f}% | ret={r['Total_Return_Pct']:+6.2f}% | MDD={r['Max_Drawdown_Pct']:5.2f}%")
    print("-" * 80)
    print(f" Total Trades  : {overall['Trades']}")
    print(f" Win Rate      : {overall['Win_Rate_Pct']}%")
    print(f" Avg Trade     : {overall['Avg_Trade_Pct']}%")
    print(f" Total Return  : {overall['Total_Return_Pct']}%")
    print(f" Max Drawdown  : {overall['Max_Drawdown_Pct']}%")
    print(f" Net PnL       : ${overall.get('Net_PnL', 0):,.0f}")
    print("=" * 80)

# ==========================================
# 12. Utilities (Wikipedia Fix)
# ==========================================
def fetch_sp500():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            tables = pd.read_html(response.read())
        return tables[0]["Symbol"].str.replace('.', '-', regex=False).tolist()
    except:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]

# ==========================================
# 13. Main
# ==========================================
if __name__ == "__main__":
    cfg = BacktestConfig()
    tickers = fetch_sp500()
    
    cands, acc, eq, yearly, monthly, overall = run_backtest_engine(tickers, cfg)
    
    save_outputs(cands, acc, eq, yearly, monthly, overall, cfg)
    print_final_report(overall, yearly)
