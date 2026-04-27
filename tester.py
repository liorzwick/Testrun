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
    max_alloc_pct: float = 0.12        
    max_positions: int = 8             
    max_portfolio_heat: float = 0.04   
    cooldown_days: int = 15
    slippage_bps: float = 12
    commission_bps: float = 2
    
    breakout_volume_ratio: float = 1.1 
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
    
    early_exit_bars: int = 15          
    early_exit_min_progress: float = -0.04 
    time_stop_bars: int = 45           
    min_profit_after_time_stop: float = 0.01 
    max_hold_bars: int = 120           
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
    cache_file = cache_dir / f"{ticker}_{start_fetch}_{end_fetch}.pkl"
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
# 2. Universe
# ==========================================
def load_universe_membership(path: str | None): return None
def ticker_allowed_on_date(ticker, dt, universe_df): return True

# ==========================================
# 3. Filters
# ==========================================
def market_filter_ok(spy_df: pd.DataFrame, current_date: pd.Timestamp) -> bool:
    x = spy_df[spy_df.index <= current_date]
    if len(x) < 220: return False
    last_row = x.iloc[-1]
    # תיקון השגיאה: השוואת ערכים סקלריים (מספרים) ולא אובייקטים של פנדס
    return float(last_row["Close"]) > float(last_row["SMA_200"])

def stock_filter_ok(today: pd.Series, cfg: BacktestConfig) -> bool:
    if any(pd.isna(today[c]) for c in ["SMA_200", "Vol_50", "High_252"]): return False
    if float(today["Close"]) < cfg.min_price or float(today["DollarVol_50"]) < cfg.min_dollar_vol_50: return False
    dist_52w = (float(today["Close"]) / float(today["High_252"])) - 1.0
    return dist_52w >= -cfg.max_dist_from_52w_high

# ==========================================
# 4. Pattern Detection (VCP / Higher Lows)
# ==========================================
def get_vcp_signal(pattern_data: pd.DataFrame, cfg: BacktestConfig):
    recent = pattern_data.tail(250)
    rim_idx = recent["High"].iloc[:-10].idxmax()
    rim_price = float(recent.loc[rim_idx, "High"])
    rim_pos = recent.index.get_loc(rim_idx)
    if rim_pos < 50: return None

    cup_bottom = float(recent["Low"].iloc[rim_pos:].min())
    cup_bottom_pos = recent.index.get_loc(recent["Low"].iloc[rim_pos:].idxmin())
    handle_area = recent.iloc[cup_bottom_pos:]
    if len(handle_area) < cfg.min_handle_days: return None
    
    handle_low = float(handle_area["Low"].min())
    if handle_low <= cup_bottom * 1.015: return None 
    
    handle_depth = (rim_price - handle_low) / rim_price
    if handle_depth > cfg.max_handle_depth: return None
    
    pivot = max(rim_price, float(handle_area["High"].max()))
    return {"pivot_price": pivot, "handle_low": handle_low}

# ==========================================
# 5. Patient Trade Simulation
# ==========================================
def simulate_trade(df: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float, initial_stop: float, cfg: BacktestConfig):
    future = df[df.index >= entry_date].head(cfg.max_hold_bars)
    if future.empty: return None
    highest_seen, stop_price = float(entry_price), float(initial_stop)
    
    for i, row in enumerate(future.itertuples()):
        highest_seen = max(highest_seen, float(row.High))
        profit = (highest_seen / entry_price) - 1
        if profit >= 0.10: stop_price = max(stop_price, entry_price * 1.01)
        if profit >= 0.20: stop_price = max(stop_price, highest_seen * 0.88)

        if float(row.Low) <= stop_price:
            exit_p = min(float(row.Open), stop_price) * (1 - cfg.slippage_bps/10000)
            return {"Exit_Date": row.Index, "Exit_Price": exit_p, "Exit_Reason": "StopHit", "Hold": i+1}
        if i >= cfg.time_stop_bars and (float(row.Close)/entry_price-1) < cfg.min_profit_after_time_stop:
            return {"Exit_Date": row.Index, "Exit_Price": float(row.Close), "Exit_Reason": "TimeExit", "Hold": i+1}
        if i == cfg.early_exit_bars and (float(row.Close)/entry_price-1) < cfg.early_exit_min_progress:
            return {"Exit_Date": row.Index, "Exit_Price": float(row.Close), "Exit_Reason": "EarlyFail", "Hold": i+1}
    return {"Exit_Date": future.index[-1], "Exit_Price": float(future.iloc[-1].Close), "Exit_Reason": "MaxHold", "Hold": len(future)}

# ==========================================
# 6. Candidate Generation
# ==========================================
def generate_candidates(tickers, data_cache, spy_df, cfg):
    candidates = []
    for ticker in tqdm(tickers, desc="Scanning"):
        df = data_cache.get(ticker, pd.DataFrame())
        if df.empty or len(df) < 250: continue
        for i in range(250, len(df)):
            past = df.iloc[:i]
            today = past.iloc[-1]
            if today.name.year < cfg.start_year or not market_filter_ok(spy_df, today.name): continue
            if not stock_filter_ok(today, cfg): continue
            pattern = get_vcp_signal(past, cfg)
            if pattern and float(today.Close) > pattern['pivot_price'] and float(past.iloc[-2].Close) <= pattern['pivot_price']:
                if float(today.Close) <= pattern['pivot_price'] * (1+cfg.max_pivot_extension):
                    if i+1 >= len(df): continue
                    entry_p = float(df.iloc[i+1].Open) * (1+cfg.slippage_bps/10000)
                    sim = simulate_trade(df, df.index[i+1], entry_p, max(pattern['handle_low'], entry_p*0.93), cfg)
                    if sim:
                        candidates.append({"Year": today.name.year, "Ticker": ticker, "Entry_Date": df.index[i+1], "Exit_Date": sim["Exit_Date"], "Entry_Price": entry_p, "Exit_Price": sim["Exit_Price"], "Pct": (sim["Exit_Price"]/entry_p-1)*100, "Reason": sim["Exit_Reason"]})
    return pd.DataFrame(candidates)

# ==========================================
# 7. Portfolio Management (REAL MONEY)
# ==========================================
def accept_trades(candidates, data_cache, cfg):
    if candidates.empty: return pd.DataFrame()
    cash, active, accepted = cfg.initial_capital, [], []
    candidates = candidates.sort_values("Entry_Date")
    for cand in candidates.to_dict("records"):
        dt = pd.Timestamp(cand["Entry_Date"])
        # שחרור הון
        for p in active[:]:
            if pd.Timestamp(p["Exit_Date"]) < dt:
                cash += p["Shares"] * p["Exit_Price"] * (1 - cfg.commission_bps/10000)
                active.remove(p)
        if len(active) >= cfg.max_positions: continue
        # חישוב הון זמין כולל פוזיציות פתוחות
        equity = cash + sum(float(data_cache[p["Ticker"]].loc[data_cache[p["Ticker"]].index <= dt].iloc[-1].Close) * p["Shares"] for p in active)
        shares = int(min(equity * cfg.max_alloc_pct, cash) / cand["Entry_Price"])
        if shares > 0:
            cash -= shares * cand["Entry_Price"] * (1 + cfg.commission_bps/10000)
            cand["Shares"] = shares
            accepted.append(cand)
            active.append(cand)
    return pd.DataFrame(accepted)

# ==========================================
# 8-13. Utils & Reporting
# ==========================================
def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        tables = pd.read_html(response.read())
    return tables[0]["Symbol"].str.replace('.', '-', regex=False).tolist()

if __name__ == "__main__":
    cfg = BacktestConfig()
    tickers = fetch_sp500()
    spy = get_data("SPY", "2014-01-01", "2026-03-01", cfg)
    # הרצה על מדגם של 200 מניות כדי למנוע קריסת זיכרון
    data_cache = {t: get_data(t, "2014-01-01", "2026-03-01", cfg) for t in tqdm(tickers[:200], desc="Loading Data")}
    data_cache["SPY"] = spy
    
    candidates = generate_candidates(tickers[:200], data_cache, spy, cfg)
    accepted = accept_trades(candidates, data_cache, cfg)
    
    if not accepted.empty:
        accepted.to_csv("accepted_trades.csv", index=False)
        print(f"\nFinal Report:\nTotal Trades: {len(accepted)}")
        print(f"Win Rate: {(accepted['Pct'] > 0).mean()*100:.1f}%")
        print(f"Avg Trade: {accepted['Pct'].mean():.2f}%")
