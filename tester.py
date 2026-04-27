import pandas as pd
import yfinance as yf
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings("ignore")

@dataclass
class BacktestConfig:
    start_year: int = 2015
    end_year: int = 2025
    benchmark: str = "SPY"
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.005      # חצי אחוז סיכון מהתיק לעסקה
    max_alloc_pct: float = 0.12        # מקסימום 12% מהתיק למניה
    max_positions: int = 8             
    max_portfolio_heat: float = 0.04   
    cooldown_days: int = 15
    slippage_bps: float = 12
    commission_bps: float = 2
    breakout_volume_ratio: float = 1.1 
    min_dollar_vol_50: float = 25_000_000
    min_price: float = 15.0
    min_risk_pct: float = 0.02
    max_risk_pct: float = 0.08         # הגדלנו מעט כדי לא לחנוק סטופים רחבים
    max_hold_bars: int = 120           # חצי שנה
    time_stop_bars: int = 45           # סבלנות של חודשיים
    min_profit_after_time_stop: float = 0.01 
    min_prior_uptrend: float = 0.08    
    min_cup_depth: float = 0.05        
    max_cup_depth: float = 0.35        
    max_handle_depth: float = 0.15     
    min_handle_days: int = 4           
    max_handle_days: int = 25
    min_cup_days: int = 20           
    max_cup_days: int = 200
    max_pivot_extension: float = 0.05  
    max_entry_extension: float = 0.04  
    max_gap_above_pivot: float = 0.03
    min_breakout_close_strength: float = 0.30
    min_rs_65: float = 0.00
    max_dist_from_52w_high: float = 0.15
    early_exit_bars: int = 15          # 3 שבועות חסד
    early_exit_min_progress: float = -0.04 
    min_tight_closes_in_handle: int = 0
    use_point_in_time_universe: bool = False
    raw_price_mode: bool = False
    allow_same_day_cash_reuse: bool = False
    universe_file: str | None = None
    output_prefix: str = "vcp_patient_v8"

# ==========================================
# 1. Data & Indicators
# ==========================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df["SMA_21"] = df["Close"].rolling(21).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_150"] = df["Close"].rolling(150).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Vol_10"] = df["Volume"].rolling(10).mean()
    df["Vol_50"] = df["Volume"].rolling(50).mean()
    df["DollarVol_50"] = df["Close"].rolling(50).mean() * df["Volume"].rolling(50).mean()
    df["Prev_Close"] = df["Close"].shift(1)
    df["ROC_20"] = df["Close"].pct_change(20)
    df["ROC_65"] = df["Close"].pct_change(65)
    df["High_252"] = df["High"].rolling(252).max()
    df["Low_10"] = df["Low"].rolling(10).min()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Prev_Close"]).abs(), (df["Low"] - df["Prev_Close"]).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"]
    return df

def get_data(ticker: str, start_fetch: str, end_fetch: str, cfg: BacktestConfig, retries: int = 3) -> pd.DataFrame:
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    price_tag = "raw" if cfg.raw_price_mode else "adj"
    cache_file = cache_dir / f"{ticker}_{start_fetch}_{end_fetch}_{price_tag}.pkl"
    if cache_file.exists(): return pd.read_pickle(cache_file)
    for _ in range(retries):
        try:
            df = yf.Ticker(ticker).history(start=start_fetch, end=end_fetch, auto_adjust=not cfg.raw_price_mode, actions=False)
            if not df.empty:
                df = add_indicators(df)
                df.to_pickle(cache_file)
                return df
            return df
        except: time.sleep(1.5)
    return pd.DataFrame()

# ==========================================
# 2. Universe Handle
# ==========================================
def load_universe_membership(path: str | None) -> pd.DataFrame | None:
    if path is None: return None
    p = Path(path)
    if not p.exists(): return None
    u = pd.read_csv(p)
    return u

def ticker_allowed_on_date(ticker: str, dt: pd.Timestamp, universe_df: pd.DataFrame | None) -> bool:
    return True

def get_sector_for_ticker(ticker: str, dt: pd.Timestamp, universe_df: pd.DataFrame | None) -> str:
    return "UNKNOWN"

# ==========================================
# 3. Filters
# ==========================================
def market_filter_ok(spy_df: pd.DataFrame, current_date: pd.Timestamp) -> bool:
    x = spy_df[spy_df.index <= current_date]
    if len(x) < 220: return False
    row = x.iloc[-1]
    sma200_old = x["SMA_200"].iloc[-20]
    if any(pd.isna(row[c]) for c in ["SMA_50", "SMA_150", "SMA_200", "ROC_65"]) or pd.isna(sma200_old): return False
    return row["Close"] > row["SMA_200"] and row["SMA_200"] > sma200_old

def stock_filter_ok(today: pd.Series, cfg: BacktestConfig) -> bool:
    required = ["SMA_21", "SMA_50", "SMA_150", "SMA_200", "Vol_50", "ATR_14", "ROC_65", "DollarVol_50", "High_252"]
    if any(pd.isna(today[c]) for c in required): return False
    if today["Close"] < cfg.min_price or today["DollarVol_50"] < cfg.min_dollar_vol_50: return False
    dist_52w = (today["Close"] / today["High_252"]) - 1.0
    if dist_52w < -cfg.max_dist_from_52w_high: return False
    return True

# ==========================================
# 4. Pattern Detection (VCP / Higher Lows)
# ==========================================
def get_cup_handle_signal(pattern_data: pd.DataFrame, cfg: BacktestConfig):
    recent = pattern_data.tail(250).copy()
    if len(recent) < 200: return None
    rim_idx = recent["High"].iloc[:-10].idxmax()
    rim_price = recent.loc[rim_idx, "High"]
    rim_pos = recent.index.get_loc(rim_idx)
    if rim_pos < 50: return None
    
    cup_bottom = recent["Low"].iloc[rim_pos:].min()
    cup_bottom_pos = recent.index.get_loc(recent["Low"].iloc[rim_pos:].idxmin())
    handle_area = recent.iloc[cup_bottom_pos:]
    if len(handle_area) < cfg.min_handle_days: return None
    
    handle_low = handle_area["Low"].min()
    # --- תנאי שפלים עולים ---
    if handle_low <= cup_bottom * 1.015: return None 
    
    handle_depth_pct = (rim_price - handle_low) / rim_price
    if handle_depth_pct > cfg.max_handle_depth: return None
    pivot = max(rim_price, handle_area["High"].max())
    
    score = (rim_price / recent["Low"].iloc[max(0, rim_pos-60)] - 1) + (1 - handle_depth_pct)
    return {"pivot_price": pivot, "handle_low": handle_low, "cup_depth_pct": (rim_price - cup_bottom) / rim_price, "handle_depth_pct": handle_depth_pct, "score": score}

# ==========================================
# 5. Trade Simulation (Patient)
# ==========================================
def simulate_trade(df: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float, initial_stop: float, initial_atr: float, pivot: float, cfg: BacktestConfig):
    future = df[df.index >= entry_date].head(cfg.max_hold_bars)
    if future.empty: return None
    highest_seen, stop_price = entry_price, initial_stop
    mfe, mae = 0.0, 0.0
    
    for i, row in enumerate(future.itertuples()):
        highest_seen = max(highest_seen, row.High)
        mfe = max(mfe, (highest_seen/entry_price-1)*100)
        mae = min(mae, (row.Low/entry_price-1)*100)
        profit_pct = (highest_seen / entry_price) - 1
        
        # מנגנון סבלני: סטופ נגרר רק אחרי רווח
        if profit_pct >= 0.10: stop_price = max(stop_price, entry_price * 1.01)
        if profit_pct >= 0.20: stop_price = max(stop_price, highest_seen * 0.88)

        if row.Low <= stop_price:
            exit_p = min(row.Open, stop_price) * (1 - cfg.slippage_bps/10000)
            return {"Exit_Date": row.Index, "Exit_Price": exit_p, "Exit_Reason": "StopHit", "Pct": (exit_p/entry_price-1)*100, "MFE": mfe, "MAE": mae, "Hold": i+1}
        if i >= cfg.time_stop_bars and (row.Close/entry_price-1) < cfg.min_profit_after_time_stop:
            return {"Exit_Date": row.Index, "Exit_Price": row.Close, "Exit_Reason": "TimeExit", "Pct": (row.Close/entry_price-1)*100, "MFE": mfe, "MAE": mae, "Hold": i+1}
        if i == cfg.early_exit_bars and (row.Close/entry_price-1) < cfg.early_exit_min_progress:
            return {"Exit_Date": row.Index, "Exit_Price": row.Close, "Exit_Reason": "EarlyFail", "Pct": (row.Close/entry_price-1)*100, "MFE": mfe, "MAE": mae, "Hold": i+1}
            
    return {"Exit_Date": future.index[-1], "Exit_Price": future.iloc[-1].Close, "Exit_Reason": "MaxHold", "Pct": (future.iloc[-1].Close/entry_price-1)*100, "MFE": mfe, "MAE": mae, "Hold": len(future)}

# ==========================================
# 6. Candidate Generation
# ==========================================
def generate_candidate_trades(tickers, data_cache, spy_df, cfg: BacktestConfig):
    candidates = []
    for ticker in tqdm(tickers, desc="Scanning"):
        df = data_cache.get(ticker, pd.DataFrame())
        if df.empty or len(df) < 250: continue
        for i in range(250, len(df)):
            past = df.iloc[:i]
            today = past.iloc[-1]
            if today.name.year < cfg.start_year or not market_filter_ok(spy_df, today.name): continue
            if not stock_filter_ok(today, cfg): continue
            pattern = get_cup_handle_signal(past, cfg)
            if pattern and today.Close > pattern['pivot_price'] and past.iloc[-2].Close <= pattern['pivot_price']:
                if today.Close <= pattern['pivot_price'] * (1+cfg.max_pivot_extension):
                    if i+1 >= len(df): continue
                    entry_p = df.iloc[i+1].Open * (1+cfg.slippage_bps/10000)
                    stop = max(pattern['handle_low'], entry_p * 0.93)
                    sim = simulate_trade(df, df.index[i+1], entry_p, stop, today.ATR_14, pattern['pivot_price'], cfg)
                    if sim:
                        candidates.append({"Year": today.name.year, "Ticker": ticker, "Signal_Date": today.name, "Entry_Date": df.index[i+1], "Entry_Price": entry_p, "Exit_Date": sim["Exit_Date"], "Exit_Price": sim["Exit_Price"], "Pct_Change": sim["Pct"], "MFE_Pct": sim["MFE"], "MAE_Pct": sim["MAE"], "Hold_Bars": sim["Hold"], "Exit_Reason": sim["Exit_Reason"], "Risk_Pct": (entry_p-stop)/entry_p*100, "Stop_Price": stop})
    if not candidates: return pd.DataFrame()
    return pd.DataFrame(candidates).sort_values("Entry_Date").reset_index(drop=True)

# ==========================================
# 7. Portfolio Acceptance (REAL MONEY)
# ==========================================
def accept_trades_with_portfolio_rules(candidates, data_cache, cfg: BacktestConfig):
    if candidates.empty: return pd.DataFrame()
    cash, active, accepted = cfg.initial_capital, [], []
    for cand in candidates.to_dict("records"):
        dt = pd.Timestamp(cand["Entry_Date"])
        # שחרור מזומן מעסקאות שנסגרו
        closed = [p for p in active if pd.Timestamp(p["Exit_Date"]) < dt]
        for p in closed: cash += p["Shares"] * p["Exit_Price"] * (1 - cfg.commission_bps/10000)
        active = [p for p in active if pd.Timestamp(p["Exit_Date"]) >= dt]
        
        if len(active) >= cfg.max_positions: continue
        equity = cash + sum(get_close_on_or_before(data_cache[p["Ticker"]], dt, p["Entry_Price"]) * p["Shares"] for p in active)
        
        shares = int(min(equity * cfg.max_alloc_pct, cash) / cand["Entry_Price"])
        if shares < 1: continue
        
        cash -= shares * cand["Entry_Price"] * (1 + cfg.commission_bps/10000)
        cand["Shares"] = shares
        cand["Net_PnL"] = (cand["Exit_Price"] - cand["Entry_Price"]) * shares
        accepted.append(cand)
        active.append(cand)
    return pd.DataFrame(accepted)

# ==========================================
# 8-13. Utils, Summaries, Reports
# ==========================================
def get_close_on_or_before(df, dt, fb):
    x = df[df.index <= dt]
    return x.iloc[-1].Close if not x.empty else fb

def build_daily_equity_curve(accepted, spy, cfg):
    if accepted.empty: return pd.DataFrame()
    dates = spy[spy.index.year >= cfg.start_year].index
    rows, cash, active = [], cfg.initial_capital, []
    for dt in dates:
        # עדכון עסקאות שנסגרו היום
        exits = accepted[accepted["Exit_Date"] == dt]
        for _, r in exits.iterrows(): cash += r["Shares"] * r["Exit_Price"]
        # עדכון עסקאות שנכנסו היום
        entries = accepted[accepted["Entry_Date"] == dt]
        for _, r in entries.iterrows(): 
            cash -= r["Shares"] * r["Entry_Price"]
            active.append(r.to_dict())
        active = [p for p in active if pd.Timestamp(p["Exit_Date"]) > dt]
        mkt_val = sum(get_close_on_or_before(get_data(p["Ticker"], "2014-01-01", "2026-03-01", cfg), dt, p["Entry_Price"]) * p["Shares"] for p in active)
        rows.append({"Date": dt, "Equity": cash + mkt_val})
    return pd.DataFrame(rows)

def print_report(df, eq):
    if df.empty: return
    print("\n" + "="*40 + "\nREPORT v8 (VCP & PATIENT)\n" + "="*40)
    print(f"Total Trades: {len(df)}")
    print(f"Win Rate: {(df['Pct_Change'] > 0).mean()*100:.1f}%")
    print(f"Avg PnL: {df['Pct_Change'].mean():.2f}%")
    if not eq.empty:
        total_ret = (eq['Equity'].iloc[-1] / eq['Equity'].iloc[0] - 1) * 100
        print(f"Total Portfolio Return: {total_ret:.2f}%")

if __name__ == "__main__":
    cfg = BacktestConfig()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tickers = pd.read_html(url)[0]["Symbol"].str.replace('.', '-', regex=False).tolist()
    spy = get_data("SPY", "2014-01-01", "2026-03-01", cfg)
    data_cache = {t: get_data(t, "2014-01-01", "2026-03-01", cfg) for t in tqdm(tickers[:150], desc="Loading Data")} # מדגם של 150 מניות
    data_cache["SPY"] = spy
    
    candidates = generate_candidate_trades(tickers[:150], data_cache, spy, cfg)
    accepted = accept_trades_with_portfolio_rules(candidates, data_cache, cfg)
    equity = build_daily_equity_curve(accepted, spy, cfg)
    
    print_report(accepted, equity)
    if not accepted.empty: accepted.to_csv("v8_portfolio_results.csv", index=False)
