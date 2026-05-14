# -*- coding: utf-8 -*-
import pandas as pd
import yfinance as yf
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
import time
from tqdm import tqdm
import os
import json

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class BacktestConfig:
    start_year: int = 2015
    end_year: int = 2025
    benchmark: str = "SPY"
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.005      
    
    custom_tickers_file: str = "mystock.csv" 
    
    max_alloc_pct: float = 0.12        
    max_positions: int = 10            
    max_portfolio_heat: float = 0.04   
    cooldown_days: int = 15
    slippage_bps: float = 12
    commission_bps: float = 2
    
    breakout_volume_ratio: float = 1.30  
    min_dollar_vol_50: float = 15_000_000 
    min_price: float = 8.0               
    
    min_risk_pct: float = 0.01         
    max_risk_pct: float = 0.10         
    max_hold_bars: int = 150           
    time_stop_bars: int = 18           
    min_profit_after_time_stop: float = 0.015 
    
    min_rs_65: float = 0.05            
    min_breakout_close_strength: float = 0.55
    max_dist_from_52w_high: float = 0.40 
    
    max_pivot_extension: float = 0.04  
    max_entry_extension: float = 0.04  
    max_gap_above_pivot: float = 0.02
    
    early_exit_bars: int = 10          
    early_exit_min_progress: float = -0.02 
    
    use_point_in_time_universe: bool = False
    raw_price_mode: bool = False
    allow_same_day_cash_reuse: bool = False
    universe_file: str | None = None
    output_prefix: str = "canslim_v31_multi_pattern"

# ==========================================
# 2. Data & Indicators
# ==========================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    df["SMA_21"] = df["Close"].rolling(21).mean()
    df["SMA_50"] = df["Close"].rolling(50, min_periods=25).mean()
    df["SMA_150"] = df["Close"].rolling(150, min_periods=75).mean()
    df["SMA_200"] = df["Close"].rolling(200, min_periods=100).mean()
    df["Vol_10"] = df["Volume"].rolling(10).mean()
    df["Vol_50"] = df["Volume"].rolling(50, min_periods=25).mean()

    df["DollarVol_50"] = df["Close"].rolling(50, min_periods=25).mean() * df["Vol_50"]
    df["Prev_Close"] = df["Close"].shift(1)
    df["ROC_20"] = df["Close"].pct_change(20)
    df["ROC_65"] = df["Close"].pct_change(65)
    df["High_252"] = df["High"].rolling(252, min_periods=120).max()
    df["Low_10"] = df["Low"].rolling(10).min()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Prev_Close"]).abs(),
        (df["Low"] - df["Prev_Close"]).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14, min_periods=7).mean()
    df["ATR_Pct"] = df["ATR_14"] / np.where(df["Close"] > 0, df["Close"], 1e-9)

    return df

def get_data(ticker: str, start_fetch: str, end_fetch: str, cfg: BacktestConfig, retries: int = 3) -> pd.DataFrame:
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    price_tag = "raw" if cfg.raw_price_mode else "adj"
    cache_file = cache_dir / f"{ticker}_{start_fetch}_{end_fetch}_{price_tag}_v31.pkl"

    if cache_file.exists():
        try: return pd.read_pickle(cache_file)
        except: pass

    for _ in range(retries):
        try:
            df = yf.Ticker(ticker).history(start=start_fetch, end=end_fetch, auto_adjust=not cfg.raw_price_mode, actions=False)
            if not df.empty:
                df = add_indicators(df)
                df.to_pickle(cache_file)
                return df
            return df
        except Exception:
            time.sleep(1.0)
    return pd.DataFrame()

# ==========================================
# 3. Universe & Filters
# ==========================================
def market_filter_ok(spy_df: pd.DataFrame, current_date: pd.Timestamp) -> bool:
    x = spy_df[spy_df.index <= current_date]
    if len(x) < 220: return False
    row = x.iloc[-1]
    if pd.isna(row["SMA_200"]) or pd.isna(row["SMA_21"]): return False
    return float(row["Close"]) > float(row["SMA_200"]) and float(row["Close"]) > float(row["SMA_21"])

def stock_filter_ok(today: pd.Series, cfg: BacktestConfig) -> bool:
    required = ["SMA_21", "SMA_50", "SMA_150", "SMA_200", "Vol_50", "ATR_14", "ATR_Pct", "ROC_65", "DollarVol_50", "High_252"]
    for c in required:
        if pd.isna(today[c]).any() if isinstance(today[c], pd.Series) else pd.isna(today[c]): return False
    
    if float(today["Close"]) < cfg.min_price: return False
    if float(today["DollarVol_50"]) < cfg.min_dollar_vol_50: return False
    if float(today["Close"]) <= float(today["SMA_50"]): return False
    
    dist_52w = (float(today["Close"]) / max(float(today["High_252"]), 1e-9)) - 1.0
    if dist_52w < -cfg.max_dist_from_52w_high: return False
    return True

# ==========================================
# 4. Pattern Detection (Multi Pattern)
# ==========================================
def get_cup_and_handle(highs, lows, vols, closes, n):
    if n < 60: return None
    recent_highs = highs[:-20]
    if len(recent_highs) == 0: return None
    left_lip_idx = int(np.argmax(recent_highs))
    left_lip_val = float(recent_highs[left_lip_idx])

    if left_lip_idx > len(recent_highs) - 10: return None 
    cup_low_idx = left_lip_idx + int(np.argmin(lows[left_lip_idx: -5]))
    cup_low_val = float(lows[cup_low_idx])

    cup_depth = (left_lip_val - cup_low_val) / max(left_lip_val, 1e-9)
    if cup_depth < 0.12 or cup_depth > 0.45: return None 

    right_side_highs = highs[cup_low_idx : -2]
    if len(right_side_highs) == 0: return None
    right_lip_idx = cup_low_idx + int(np.argmax(right_side_highs))
    right_lip_val = float(highs[right_lip_idx])

    if abs(left_lip_val - right_lip_val) / max(left_lip_val, 1e-9) > 0.12: return None 
    pivot = max(left_lip_val, right_lip_val)

    handle_len = (len(highs) - 1) - right_lip_idx
    if handle_len < 3 or handle_len > 60: return None 

    handle_low = float(np.min(lows[right_lip_idx:]))
    handle_depth = (right_lip_val - handle_low) / max(right_lip_val, 1e-9)

    if handle_depth > cup_depth * 0.5: return None 
    if handle_low < cup_low_val + (pivot - cup_low_val) * 0.4: return None 

    return {"type": "Cup & Handle", "pivot_price": pivot, "tight_low": handle_low, "base_depth": cup_depth, "tightness": handle_depth, "base_length": len(highs) - left_lip_idx}

def get_bull_flag(highs, lows, vols, closes, n):
    if n < 40: return None
    search_window = highs[-90:-4] 
    if len(search_window) == 0: return None

    pole_peak_idx_local = int(np.argmax(search_window))
    absolute_peak_idx = (n - 90) + pole_peak_idx_local
    pole_peak_val = float(highs[absolute_peak_idx])

    start_search_idx = max(0, absolute_peak_idx - 40)
    pole_start_val = float(np.min(lows[start_search_idx : absolute_peak_idx]))

    if (pole_peak_val - pole_start_val) / max(pole_start_val, 1e-9) < 0.18: return None 

    flag_length = (n - 1) - absolute_peak_idx
    if flag_length < 4 or flag_length > 45: return None 

    flag_lows = lows[absolute_peak_idx : -1]
    if len(flag_lows) == 0: return None
    flag_low = float(np.min(flag_lows))

    flag_depth = (pole_peak_val - flag_low) / max(pole_peak_val, 1e-9)
    if flag_depth < 0.03 or flag_depth > 0.20: return None 

    max_in_flag = float(np.max(highs[absolute_peak_idx+1 : -1]))
    if max_in_flag > pole_peak_val * 1.01: return None 

    return {"type": "Bull Flag", "pivot_price": pole_peak_val, "tight_low": flag_low, "base_depth": flag_depth, "tightness": flag_depth, "base_length": flag_length}

def get_darvas_box(highs, lows, vols, closes, n):
    box_length = 30 
    if n < box_length + 40: return None

    window_highs = highs[-box_length:-2]
    window_lows = lows[-box_length:-2]
    window_closes = closes[-box_length:-2]

    box_top = float(np.max(window_highs))
    box_bottom = float(np.min(window_lows))

    box_depth = (box_top - box_bottom) / max(box_top, 1e-9)
    if box_depth < 0.04 or box_depth > 0.12: return None 

    days_in_box = np.sum((window_closes >= box_bottom * 0.98) & (window_closes <= box_top * 1.02))
    if (days_in_box / len(window_closes)) < 0.85: return None 

    top_touches = np.where(window_highs >= box_top * 0.985)[0]
    if len(top_touches) < 2: return None
    if top_touches[-1] - top_touches[0] < 12: return None 

    return {"type": "Darvas Box", "pivot_price": box_top, "tight_low": box_bottom, "base_depth": box_depth, "tightness": box_depth, "base_length": box_length}

def get_double_bottom(highs, lows, vols, n):
    if n < 100: return None
    recent_lows = lows[-300:-20]
    if len(recent_lows) == 0: return None
    left_bottom_idx = int(np.argmin(recent_lows))
    left_bottom_val = float(recent_lows[left_bottom_idx])

    # השיא שבין שני השפלים - זוהי נקודת הפיבוט הבלעדית!
    mid_section = highs[left_bottom_idx : -5]
    if len(mid_section) < 10: return None
    mid_peak_idx = left_bottom_idx + int(np.argmax(mid_section))
    mid_peak_val = float(highs[mid_peak_idx])

    right_section = lows[mid_peak_idx : -1]
    if len(right_section) < 5: return None
    right_bottom_idx = mid_peak_idx + int(np.argmin(right_section))
    right_bottom_val = float(lows[right_bottom_idx])

    if abs(left_bottom_val - right_bottom_val) / max(left_bottom_val, 1e-9) > 0.08: return None 

    base_depth = (mid_peak_val - min(left_bottom_val, right_bottom_val)) / max(mid_peak_val, 1e-9)
    if base_depth < 0.10 or base_depth > 0.40: return None 

    # הפיבוט הוא אך ורק נקודת האמצע. אין ידית בדאבל בוטום קלאסי!
    pivot = mid_peak_val
    tight_low = float(np.min(lows[right_bottom_idx:]))
    tightness = (pivot - tight_low) / max(pivot, 1e-9)

    return {"type": "Double Bottom", "pivot_price": pivot, "tight_low": tight_low, "base_depth": base_depth, "tightness": tightness, "base_length": right_bottom_idx - left_bottom_idx}

def check_classical_patterns(hist):
    hist_filtered = hist.dropna(subset=['High', 'Low', 'Volume', 'Close'])
    if len(hist_filtered) < 60: return None

    highs = hist_filtered["High"].astype(float).values
    lows = hist_filtered["Low"].astype(float).values
    vols = hist_filtered["Volume"].astype(float).values
    closes = hist_filtered["Close"].astype(float).values
    n = len(hist_filtered)

    pattern = get_bull_flag(highs, lows, vols, closes, n)
    if pattern: return pattern

    pattern = get_cup_and_handle(highs, lows, vols, closes, n)
    if pattern: return pattern

    pattern = get_double_bottom(highs, lows, vols, n)
    if pattern: return pattern

    pattern = get_darvas_box(highs, lows, vols, closes, n)
    if pattern: return pattern

    return None

# ==========================================
# 5. Patient Trade Simulation (Scale-Out V30)
# ==========================================
def classify_pnl(pct: float) -> str:
    if pct > 0: return "Win"
    if pct < 0: return "Loss"
    return "Flat"

def simulate_trade(df: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float, initial_stop: float, cfg: BacktestConfig):
    future = df[df.index >= entry_date].head(cfg.max_hold_bars)
    if future.empty: return None

    stop_today = float(initial_stop)
    stop_next_day = float(initial_stop)
    if stop_next_day >= entry_price: stop_next_day = entry_price * 0.985 
        
    highest_seen = float(entry_price)
    lowest_seen = float(entry_price)
    
    scaled_out = False
    scale_out_price = 0.0

    for i, row in enumerate(future.itertuples()):
        dt = row.Index
        day_open = float(row.Open)
        day_high = float(row.High)
        day_low = float(row.Low)
        day_close = float(row.Close)

        stop_today = initial_stop if i == 0 else stop_next_day

        if day_open <= stop_today:
            current_exit = day_open * (1 - cfg.slippage_bps / 10000)
            blended_exit = ((scale_out_price + current_exit) / 2.0) if scaled_out else current_exit
            gross_pct = (blended_exit - entry_price) / max(entry_price, 1e-9) * 100
            net_pct = gross_pct - (2 * cfg.commission_bps / 100)
            reason = "GapStop_Scaled" if scaled_out else "GapStop"
            return {"Exit_Date": dt, "Exit_Price": blended_exit, "Exit_Reason": reason, "Hold_Bars": i+1,
                    "Pct_Change": round(net_pct, 2), "MFE_Pct": round((highest_seen/max(entry_price, 1e-9)-1)*100, 2),
                    "MAE_Pct": round((min(lowest_seen, day_open)/max(entry_price, 1e-9)-1)*100, 2)}

        if day_low <= stop_today:
            current_exit = stop_today * (1 - cfg.slippage_bps / 10000)
            blended_exit = ((scale_out_price + current_exit) / 2.0) if scaled_out else current_exit
            gross_pct = (blended_exit - entry_price) / max(entry_price, 1e-9) * 100
            net_pct = gross_pct - (2 * cfg.commission_bps / 100)
            reason = "StopHit_Scaled" if scaled_out else "StopHit"
            return {"Exit_Date": dt, "Exit_Price": blended_exit, "Exit_Reason": reason, "Hold_Bars": i+1,
                    "Pct_Change": round(net_pct, 2), "MFE_Pct": round((highest_seen/max(entry_price, 1e-9)-1)*100, 2),
                    "MAE_Pct": round((min(lowest_seen, day_low)/max(entry_price, 1e-9)-1)*100, 2)}

        highest_seen = max(highest_seen, day_high)
        lowest_seen = min(lowest_seen, day_low)
        profit_high = (highest_seen - entry_price) / max(entry_price, 1e-9)
        
        new_stop = stop_today
        if not scaled_out and profit_high >= 0.10: 
            scaled_out = True
            scale_out_price = entry_price * 1.10 
            new_stop = max(new_stop, entry_price * 1.005) 
            
        if profit_high >= 0.15: new_stop = max(new_stop, highest_seen * 0.92) 
        if profit_high >= 0.25: new_stop = max(new_stop, highest_seen * 0.88) 

        stop_next_day = max(stop_today, new_stop)

        if (i + 1) == cfg.early_exit_bars:
            if (day_close / max(entry_price, 1e-9)) - 1.0 < cfg.early_exit_min_progress:
                current_exit = day_close * (1 - cfg.slippage_bps / 10000)
                blended_exit = ((scale_out_price + current_exit) / 2.0) if scaled_out else current_exit
                gross_pct = (blended_exit - entry_price) / max(entry_price, 1e-9) * 100
                net_pct = gross_pct - (2 * cfg.commission_bps / 100)
                reason = "EarlyFail_Scaled" if scaled_out else "EarlyFail"
                return {"Exit_Date": dt, "Exit_Price": blended_exit, "Exit_Reason": reason, "Hold_Bars": i+1,
                        "Pct_Change": round(net_pct, 2), "MFE_Pct": round((highest_seen/max(entry_price, 1e-9)-1)*100, 2),
                        "MAE_Pct": round((lowest_seen/max(entry_price, 1e-9)-1)*100, 2)}

        if (i + 1) >= cfg.time_stop_bars:
            if (day_close - entry_price) / max(entry_price, 1e-9) < cfg.min_profit_after_time_stop:
                current_exit = day_close * (1 - cfg.slippage_bps / 10000)
                blended_exit = ((scale_out_price + current_exit) / 2.0) if scaled_out else current_exit
                gross_pct = (blended_exit - entry_price) / max(entry_price, 1e-9) * 100
                net_pct = gross_pct - (2 * cfg.commission_bps / 100)
                reason = "TimeExit_Scaled" if scaled_out else "TimeExit"
                return {"Exit_Date": dt, "Exit_Price": blended_exit, "Exit_Reason": reason, "Hold_Bars": i+1,
                        "Pct_Change": round(net_pct, 2), "MFE_Pct": round((highest_seen/max(entry_price, 1e-9)-1)*100, 2),
                        "MAE_Pct": round((lowest_seen/max(entry_price, 1e-9)-1)*100, 2)}

    current_exit = float(future.iloc[-1]["Close"]) * (1 - cfg.slippage_bps / 10000)
    blended_exit = ((scale_out_price + current_exit) / 2.0) if scaled_out else current_exit
    gross_pct = (blended_exit - entry_price) / max(entry_price, 1e-9) * 100
    net_pct = gross_pct - (2 * cfg.commission_bps / 100)
    reason = "MaxHold_Scaled" if scaled_out else "MaxHold"
    return {"Exit_Date": future.index[-1], "Exit_Price": blended_exit, "Exit_Reason": reason, "Hold_Bars": len(future),
            "Pct_Change": round(net_pct, 2), "MFE_Pct": round((highest_seen/max(entry_price, 1e-9)-1)*100, 2),
            "MAE_Pct": round((lowest_seen/max(entry_price, 1e-9)-1)*100, 2)}

# ==========================================
# 6. Candidate Generation
# ==========================================
def generate_candidate_trades(tickers, data_cache, spy_df, cfg: BacktestConfig):
    candidates = []
    print(f"\nScanning {len(tickers)} stocks across ALL Classical Patterns...")

    for year in tqdm(range(cfg.start_year, cfg.end_year + 1), desc="Years"):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        for ticker in tickers:
            try:
                df = data_cache.get(ticker, pd.DataFrame())
                if df.empty or len(df) < 260: continue

                test_days = df[(df.index >= test_start) & (df.index <= test_end)].index
                for current_date in test_days:
                    if not market_filter_ok(spy_df, current_date): continue

                    past_data = df[df.index <= current_date]
                    if len(past_data) < 251: continue

                    today = past_data.iloc[-1]
                    yesterday = past_data.iloc[-2]
                    lookback_data = past_data.iloc[:-1].copy()

                    if not stock_filter_ok(today, cfg): continue
                    
                    spy_past = spy_df[spy_df.index <= current_date]
                    if not spy_past.empty:
                        spy_rs = float(spy_past.iloc[-1]["ROC_65"])
                        stock_rs = float(today["ROC_65"])
                        if (stock_rs - spy_rs) < cfg.min_rs_65: continue

                    pattern = check_classical_patterns(lookback_data)
                    if pattern is None: continue

                    pivot = pattern["pivot_price"]
                    prev_close = float(yesterday["Close"])
                    close = float(today["Close"])
                    
                    # הפריצה עצמה: מחיר חייב לעבור את הפיבוט בדיוק היום
                    if not (prev_close <= pivot and close > pivot): continue

                    day_range = max(float(today["High"]) - float(today["Low"]), 1e-9)
                    close_strength = (close - float(today["Low"])) / day_range
                    if close_strength < cfg.min_breakout_close_strength: continue

                    vol_ratio = float(today["Volume"]) / max(float(today["Vol_50"]), 1e-9)
                    if vol_ratio < cfg.breakout_volume_ratio: continue

                    next_bar = df[df.index > current_date].head(1)
                    if next_bar.empty: continue

                    entry_date = next_bar.index[0]
                    entry_open = float(next_bar.iloc[0]["Open"])

                    gap_from_pivot = (entry_open / max(pivot, 1e-9)) - 1.0
                    if gap_from_pivot > cfg.max_gap_above_pivot: continue

                    entry_price = entry_open * (1 + cfg.slippage_bps / 10000)
                    if entry_price > pivot * (1 + cfg.max_entry_extension): continue

                    atr = float(today["ATR_14"])
                    tight_low = float(pattern["tight_low"])
                    calculated_stop = tight_low - (0.5 * atr) 
                    max_allowed_stop = entry_price * (1 - cfg.max_risk_pct)
                    initial_stop = max(calculated_stop, max_allowed_stop)
                    
                    risk_pct = (entry_price - initial_stop) / max(entry_price, 1e-9)
                    if not (cfg.min_risk_pct <= risk_pct <= cfg.max_risk_pct): continue

                    sim = simulate_trade(df, entry_date, entry_price, initial_stop, cfg)
                    if sim is None: continue

                    candidates.append({
                        "Year": year, "Ticker": ticker, "Pattern_Type": pattern["type"],
                        "Signal_Date": current_date, "Entry_Date": entry_date, "Entry_Price": round(entry_price, 2),
                        "Exit_Date": sim["Exit_Date"], "Exit_Price": round(sim["Exit_Price"], 2), "Pct_Change": sim["Pct_Change"],
                        "Risk_Pct": round(risk_pct * 100, 2), "Stop_Price": round(initial_stop, 2),
                        "Base_Depth_Pct": round(pattern["base_depth"] * 100, 2), "Tightness_Pct": round(pattern["tightness"] * 100, 2),
                        "Base_Length": int(pattern["base_length"]), 
                        "Volume_Ratio": round(vol_ratio, 2), "RS_65": round(stock_rs, 4), 
                        "Close_Strength": round(close_strength, 4), "Gap_From_Pivot": round(gap_from_pivot, 4),
                        "Hold_Bars": sim["Hold_Bars"], "Result": classify_pnl(sim["Pct_Change"]),
                        "Exit_Reason": sim["Exit_Reason"], "MFE_Pct": sim["MFE_Pct"], "MAE_Pct": sim["MAE_Pct"]
                    })
            except Exception as e:
                pass

    if not candidates: return pd.DataFrame()
    return pd.DataFrame(candidates).sort_values(["Entry_Date", "Volume_Ratio"], ascending=[True, False]).reset_index(drop=True)

# ==========================================
# 7. Portfolio Management
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
        current_heat = sum(float(pos.get("Risk_Dollars", 0.0)) for pos in active)
        max_heat = equity * cfg.max_portfolio_heat
        remaining_heat = max(0.0, max_heat - current_heat)

        if remaining_heat <= 0: continue

        shares_by_risk = min(max_risk_dollars_trade, remaining_heat) / risk_per_share
        shares_by_alloc = (equity * cfg.max_alloc_pct) / max(entry_price, 1e-9)
        shares_by_cash = cash / (max(entry_price, 1e-9) * (1 + cfg.commission_bps / 10000))

        shares = int(np.floor(min(shares_by_risk, shares_by_alloc, shares_by_cash)))
        if shares < 1: continue

        entry_fee = shares * entry_price * cfg.commission_bps / 10000
        exit_fee = shares * exit_price * cfg.commission_bps / 10000
        total_cost = shares * entry_price + entry_fee
        if total_cost > cash: continue

        cash -= total_cost

        t = cand.copy()
        t["Shares"] = shares
        t["Entry_Fee"] = round(entry_fee, 2)
        t["Exit_Fee"] = round(exit_fee, 2)
        t["Gross_Entry"] = round(shares * entry_price, 2)
        t["Gross_Exit"] = round(shares * exit_price, 2)
        t["Net_PnL"] = round((shares * (exit_price - entry_price)) - entry_fee - exit_fee, 2)
        t["Alloc_Pct"] = round(shares * entry_price / max(equity, 1e-9) * 100, 2) if equity > 0 else 0.0
        t["Risk_Dollars"] = round(shares * risk_per_share, 2)
        
        accepted.append(t)
        last_exit_by_ticker[ticker] = pd.Timestamp(t["Exit_Date"])
        active.append({
            "Ticker": ticker, "Entry_Date": t["Entry_Date"], "Exit_Date": t["Exit_Date"],
            "Entry_Price": entry_price, "Exit_Price": exit_price, "Shares": shares, "Exit_Fee": exit_fee,
            "Risk_Dollars": t["Risk_Dollars"],
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

    wins = trades_df[trades_df["Pct_Change"] > 0]
    losses = trades_df[trades_df["Pct_Change"] < 0]

    total_return, max_dd = 0.0, 0.0
    if equity_df is not None and not equity_df.empty:
        total_return = round((equity_df["Equity"].iloc[-1] / max(equity_df["Equity"].iloc[0], 1e-9) - 1.0) * 100, 2)
        max_dd = calc_drawdown(equity_df["Equity"])

    return {
        "Trades": len(trades_df), "Wins": len(wins), "Losses": len(losses),
        "Win_Rate_Pct": round(len(wins) / len(trades_df) * 100, 2) if len(trades_df) > 0 else 0.0,
        "Avg_Trade_Pct": round(trades_df["Pct_Change"].mean(), 2),
        "Avg_Win_Pct": round(wins["Pct_Change"].mean(), 2) if len(wins) > 0 else 0.0,
        "Avg_Loss_Pct": round(losses["Pct_Change"].mean(), 2) if len(losses) > 0 else 0.0,
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
            Win_Rate_Pct=("Pct_Change", lambda s: round(s.gt(0).sum() / len(s) * 100, 2) if len(s) > 0 else 0),
            Avg_Trade_Pct=("Pct_Change", lambda s: round(s.mean(), 2)),
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
    
    for t in tqdm(tickers, desc="Loading Data"):
        data_cache[t] = get_data(t, "2014-01-01", "2026-03-01", cfg)
        
    cands = generate_candidate_trades(tickers, data_cache, spy, cfg)
    acc = accept_trades_with_portfolio_rules(cands, data_cache, cfg)
    eq = build_daily_equity_curve(acc, data_cache, spy, cfg)
    
    yearly_df = yearly_summary(acc, eq)
    monthly_df = monthly_summary(acc)
    overall = summarize_trades(acc, eq)
    
    return cands, acc, eq, yearly_df, monthly_df, overall

# ==========================================
# 11. Output & Case Studies Report
# ==========================================
def calculate_pattern_success(accepted_df: pd.DataFrame):
    if accepted_df.empty: return pd.DataFrame()
    stats = accepted_df.groupby("Pattern_Type").agg(
        Total_Trades=("Ticker", "count"),
        Win_Rate=("Pct_Change", lambda x: round((x > 0).sum() / len(x) * 100, 1)),
        Avg_Trade=("Pct_Change", lambda x: round(x.mean(), 2)),
        Net_PnL=("Net_PnL", "sum")
    ).reset_index().sort_values("Avg_Trade", ascending=False)
    return stats

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
    
    pattern_stats = calculate_pattern_success(accepted_df)
    if not pattern_stats.empty:
        pattern_stats.to_csv(out_dir / "pattern_performance.csv", index=False, encoding="utf-8-sig")
        
    print(f"\nFiles saved -> {out_dir}/")

def print_final_report(overall: dict, yearly_df: pd.DataFrame, accepted_df: pd.DataFrame):
    if not overall or overall.get('Trades', 0) == 0:
        print("No trades executed.")
        return
    print("\n" + "=" * 80)
    print("VCP BACKTEST REPORT (v31 - Multi-Pattern & Scale Out)")
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
    
    print("\n--- PERFORMANCE BY PATTERN TYPE ---")
    stats = calculate_pattern_success(accepted_df)
    if not stats.empty:
        print(stats.to_string(index=False))
    print("=" * 80)

    # --- יצירת ה-Case Studies (הדוגמאות) ---
    print("\n--- 📖 CASE STUDIES: עסקאות מופת לפי תבנית ---")
    if not accepted_df.empty:
        for pattern in accepted_df['Pattern_Type'].unique():
            pattern_df = accepted_df[accepted_df['Pattern_Type'] == pattern]
            if pattern_df.empty: continue
            
            # שליפת העסקה הכי רווחית לכל תבנית
            best_trade = pattern_df.sort_values(by='Pct_Change', ascending=False).iloc[0]
            
            entry_date = best_trade['Entry_Date'].strftime('%Y-%m-%d')
            exit_date = best_trade['Exit_Date'].strftime('%Y-%m-%d')
            
            print(f"\n🔥 תבנית: {pattern}")
            print(f"📌 מניה: {best_trade['Ticker']} | כניסה: {entry_date} | יציאה: {exit_date}")
            print(f"💰 רווח שמומש: {best_trade['Pct_Change']}% (רווח שיא MFE: {best_trade['MFE_Pct']}%)")
            print(f"🚪 סיבת יציאה: {best_trade['Exit_Reason']}")
            print("💡 למה הבוט נכנס? (נתוני הפריצה):")
            print(f"   - פיצוץ ווליום מוסדי: פי {best_trade['Volume_Ratio']} ממוצע 50 יום")
            print(f"   - כיווץ (Tightness): {best_trade['Tightness_Pct']}%")
            print(f"   - עומק הבסיס: {best_trade['Base_Depth_Pct']}%")
            print(f"   - אורך התבנית: {best_trade['Base_Length']} ימים")
            print(f"   - חוזק סגירה יומית: {best_trade['Close_Strength']}")
            print(f"   - עוצמה יחסית (RS): {best_trade['RS_65']}")
            print("-" * 60)

# ==========================================
# 12. Utilities
# ==========================================
def get_tickers(cfg: BacktestConfig):
    if cfg.custom_tickers_file and os.path.exists(cfg.custom_tickers_file):
        try:
            df = pd.read_csv(cfg.custom_tickers_file)
            col_name = next((c for c in df.columns if c.strip().lower() in ['ticker', 'symbol']), None)
            if col_name:
                tickers = df[col_name].dropna().astype(str).str.strip().str.upper().tolist()
                tickers = [t.replace('.', '-') for t in tickers if t.isalpha() or '-' in t or '.' in t]
                valid_tickers = sorted(list(set(tickers)))
                print(f"✅ Loaded {len(valid_tickers)} custom tickers from {cfg.custom_tickers_file}")
                return valid_tickers
            else:
                raise ValueError(f"Could not find 'Ticker' column in {cfg.custom_tickers_file}.")
        except Exception as e:
            raise RuntimeError(f"Error loading custom file: {e}")
    else:
        raise FileNotFoundError(f"Custom file '{cfg.custom_tickers_file}' not found.")

# ==========================================
# 13. Main
# ==========================================
if __name__ == "__main__":
    cfg = BacktestConfig()
    tickers = get_tickers(cfg)
    cands, acc, eq, yearly, monthly, overall = run_backtest_engine(tickers, cfg)
    save_outputs(cands, acc, eq, yearly, monthly, overall, cfg)
    print_final_report(overall, yearly, acc)
