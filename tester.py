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
from fastdtw import fastdtw

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION (Pure Alpha Sniper)
# ==========================================
@dataclass
class BacktestConfig:
    start_year: int = 2015
    end_year: int = 2025
    benchmark: str = "SPY"
    initial_capital: float = 100_000.0
    
    # ניהול הון מרוכז, קטלני ואגרסיבי למניות העל בלבד
    risk_per_trade: float = 0.015        # 1.5% סיכון לטרייד (אגרוף פלדה)
    max_alloc_pct: float = 0.25          # קונה עד 25% מהתיק למניה אחת!
    max_positions: int = 5               # מקסימום 5 פוזיציות. תיק מרוכז שמייצר תשואת יתר.
    max_portfolio_heat: float = 0.075    # 5 מניות כפול 1.5% סיכון = 7.5% מקסימום
    max_new_trades_per_day: int = 2      
    cooldown_days: int = 3               
    
    custom_tickers_file: str = "mystock.csv" 
    slippage_bps: float = 12
    commission_bps: float = 2
    
    # חוקי סינון אכזריים (איכות על פני כמות)
    breakout_volume_ratio: float = 1.30  # מוסדיים בלבד! לא יורדים מ-1.3
    min_breakout_close_strength: float = 0.55 # סגירה חזקה
    min_dollar_vol_50: float = 10_000_000 
    min_price: float = 8.0               
    
    min_risk_pct: float = 0.01         
    max_risk_pct: float = 0.12         
    max_hold_bars: int = 150           
    time_stop_bars: int = 18           
    min_profit_after_time_stop: float = 0.015 
    
    min_rs_65: float = 0.05            
    bear_market_rs_threshold: float = 0.25 
    
    max_dist_from_52w_high: float = 0.45 
    max_pivot_extension: float = 0.04  
    max_entry_extension: float = 0.04  
    max_gap_above_pivot: float = 0.02
    
    early_exit_bars: int = 10          
    early_exit_min_progress: float = -0.02 
    
    use_point_in_time_universe: bool = False
    raw_price_mode: bool = False
    allow_same_day_cash_reuse: bool = False
    universe_file: str | None = None
    output_prefix: str = "canslim_v46_pure_alpha"

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
    cache_file = cache_dir / f"{ticker}_{start_fetch}_{end_fetch}_{price_tag}_v46.pkl"

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
# 3. Filters
# ==========================================
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
# 4. Pattern Detection (STRICT DTW VISION)
# ==========================================
def normalize_series(series):
    series_array = np.array(series)
    min_val = np.min(series_array)
    max_val = np.max(series_array)
    if max_val == min_val:
        return np.zeros(len(series_array))
    return (series_array - min_val) / (max_val - min_val)

def get_dtw_templates():
    templates = {}
    
    pole = np.linspace(0, 1.0, 10)
    flag_trend = np.linspace(1.0, 0.7, 20)
    flag_waves = flag_trend + np.sin(np.linspace(0, 4*np.pi, 20)) * 0.05 
    templates["Bull Flag"] = {
        "data": np.concatenate((pole, flag_waves)), 
        "windows": list(range(15, 66, 5)), 
        "threshold": 0.12, "min_corr": 0.88, "comp": 0  # חזרה ל-88% קורלציה, אפס סובלנות לזבל
    }

    rise = np.linspace(0, 1.0, 10)
    initial_pullback = np.linspace(1.0, 0.8, 5) 
    box = np.ones(35) * 0.9 + np.sin(np.linspace(0, 6*np.pi, 35)) * 0.05
    templates["Darvas Box"] = {
        "data": np.concatenate((rise, initial_pullback, box)), 
        "windows": list(range(25, 151, 10)), 
        "threshold": 0.12, "min_corr": 0.85, "comp": 1
    }

    left_cup = np.linspace(-1, 0, 45)**2
    right_cup = np.linspace(0, 1, 20)**2
    handle_trend = np.linspace(1.0, 0.7, 15)
    handle_waves = handle_trend + np.cos(np.linspace(0, 2*np.pi, 15)) * 0.03
    templates["Cup & Handle"] = {
        "data": np.concatenate((left_cup, right_cup, handle_waves)), 
        "windows": list(range(40, 301, 15)), 
        "threshold": 0.15, "min_corr": 0.82, "comp": 2
    }
    
    return templates

def check_classical_patterns(hist):
    hist_filtered = hist.dropna(subset=['Close'])
    if len(hist_filtered) < 15: return None
    closes = hist_filtered["Close"].astype(float).values
    
    templates = get_dtw_templates()
    best_pattern = None
    best_score = float('inf')

    for name, config in templates.items():
        for window in config["windows"]:
            if len(closes) < window:
                continue
            
            current_closes = closes[-window:]
            
            raw_min = np.min(current_closes)
            raw_max = np.max(current_closes)
            if raw_min == 0 or (raw_max - raw_min) / raw_min < 0.10:
                continue
                
            norm_current = normalize_series(current_closes)
            
            if np.std(norm_current) > 0.35:
                continue

            x_orig = np.linspace(0, 1, len(config["data"]))
            x_new = np.linspace(0, 1, window)
            resized_template = np.interp(x_new, x_orig, config["data"])
            norm_template_resized = normalize_series(resized_template)
            
            corr = np.corrcoef(norm_current, norm_template_resized)[0, 1]
            
            if pd.isna(corr) or corr < config["min_corr"]:
                continue
                
            w = window
            if "Flag" in name:
                start_price = current_closes[0]
                end_price = current_closes[-1]
                if end_price <= start_price * 1.05: continue
                
            elif "Cup" in name:
                cup_bottom = np.min(current_closes[:int(w*0.8)])
                handle_bottom = np.min(current_closes[-int(w*0.2):])
                if handle_bottom < cup_bottom: continue

            distance, path = fastdtw(norm_current, norm_template_resized, dist=lambda x, y: abs(x - y))
            avg_distance = distance / window
            
            if avg_distance < config["threshold"] and avg_distance < best_score:
                best_score = avg_distance
                
                if "Flag" in name:
                    pivot = float(np.max(current_closes[:int(w*0.5)]))
                    low = float(np.min(current_closes[-int(w*0.5):]))
                elif "Darvas" in name:
                    pivot = float(np.max(current_closes[-int(w*0.7):]))
                    low = float(np.min(current_closes[-int(w*0.7):]))
                else: 
                    pivot = float(np.max(current_closes[int(w*0.6):int(w*0.9)]))
                    low = float(np.min(current_closes[-int(w*0.3):]))
                
                tightness = (pivot - low) / max(pivot, 1e-9)
                
                best_pattern = {
                    "type": name,
                    "dtw_distance": round(avg_distance, 3),
                    "correlation": round(corr * 100, 1),
                    "complexity_bonus": config["comp"],
                    "threshold": config["threshold"],
                    "pivot_price": pivot,
                    "tight_low": low,
                    "last_pullback_low": low,
                    "tightness": tightness,
                    "base_depth": 0.20, 
                    "dry_up_ratio": 1.0, 
                    "touches": 2,
                    "base_length": window
                }
                
    return best_pattern

# ==========================================
# 5. Patient Trade Simulation (WIDE RUNNERS)
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
    scale_out_pct = 0.33 

    for i, row in enumerate(future.itertuples()):
        dt = row.Index
        day_open = float(row.Open)
        day_high = float(row.High)
        day_low = float(row.Low)
        day_close = float(row.Close)

        stop_today = initial_stop if i == 0 else stop_next_day

        if day_open <= stop_today:
            current_exit = day_open * (1 - cfg.slippage_bps / 10000)
            blended_exit = (scale_out_price * scale_out_pct + current_exit * (1 - scale_out_pct)) if scaled_out else current_exit
            gross_pct = (blended_exit - entry_price) / max(entry_price, 1e-9) * 100
            net_pct = gross_pct - (2 * cfg.commission_bps / 100)
            reason = "GapStop_Scaled" if scaled_out else "GapStop"
            return {"Exit_Date": dt, "Exit_Price": blended_exit, "Exit_Reason": reason, "Hold_Bars": i+1,
                    "Pct_Change": round(net_pct, 2), "MFE_Pct": round((highest_seen/max(entry_price, 1e-9)-1)*100, 2),
                    "MAE_Pct": round((min(lowest_seen, day_open)/max(entry_price, 1e-9)-1)*100, 2)}

        if day_low <= stop_today:
            current_exit = stop_today * (1 - cfg.slippage_bps / 10000)
            blended_exit = (scale_out_price * scale_out_pct + current_exit * (1 - scale_out_pct)) if scaled_out else current_exit
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
        
        if profit_high >= 0.10: 
            new_stop = max(new_stop, entry_price * 1.01) 
            
        if not scaled_out and profit_high >= 0.20: 
            scaled_out = True
            scale_out_price = entry_price * 1.20 
            new_stop = max(new_stop, entry_price * 1.05) 
            
        if profit_high >= 0.20: 
            new_stop = max(new_stop, highest_seen * 0.85) 
        if profit_high >= 0.50: 
            new_stop = max(new_stop, highest_seen * 0.82) 

        stop_next_day = max(stop_today, new_stop)

        if (i + 1) == cfg.early_exit_bars:
            if (day_close / max(entry_price, 1e-9)) - 1.0 < cfg.early_exit_min_progress:
                current_exit = day_close * (1 - cfg.slippage_bps / 10000)
                blended_exit = (scale_out_price * scale_out_pct + current_exit * (1 - scale_out_pct)) if scaled_out else current_exit
                gross_pct = (blended_exit - entry_price) / max(entry_price, 1e-9) * 100
                net_pct = gross_pct - (2 * cfg.commission_bps / 100)
                reason = "EarlyFail_Scaled" if scaled_out else "EarlyFail"
                return {"Exit_Date": dt, "Exit_Price": blended_exit, "Exit_Reason": reason, "Hold_Bars": i+1,
                        "Pct_Change": round(net_pct, 2), "MFE_Pct": round((highest_seen/max(entry_price, 1e-9)-1)*100, 2),
                        "MAE_Pct": round((lowest_seen/max(entry_price, 1e-9)-1)*100, 2)}

        if (i + 1) >= cfg.time_stop_bars:
            if (day_close - entry_price) / max(entry_price, 1e-9) < cfg.min_profit_after_time_stop:
                current_exit = day_close * (1 - cfg.slippage_bps / 10000)
                blended_exit = (scale_out_price * scale_out_pct + current_exit * (1 - scale_out_pct)) if scaled_out else current_exit
                gross_pct = (blended_exit - entry_price) / max(entry_price, 1e-9) * 100
                net_pct = gross_pct - (2 * cfg.commission_bps / 100)
                reason = "TimeExit_Scaled" if scaled_out else "TimeExit"
                return {"Exit_Date": dt, "Exit_Price": blended_exit, "Exit_Reason": reason, "Hold_Bars": i+1,
                        "Pct_Change": round(net_pct, 2), "MFE_Pct": round((highest_seen/max(entry_price, 1e-9)-1)*100, 2),
                        "MAE_Pct": round((lowest_seen/max(entry_price, 1e-9)-1)*100, 2)}

    current_exit = float(future.iloc[-1]["Close"]) * (1 - cfg.slippage_bps / 10000)
    blended_exit = (scale_out_price * scale_out_pct + current_exit * (1 - scale_out_pct)) if scaled_out else current_exit
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
    print(f"\nScanning {len(tickers)} stocks... (V46: Pure Alpha Sniper Active)")

    for year in tqdm(range(cfg.start_year, cfg.end_year + 1), desc="Years"):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        for ticker in tickers:
            try:
                df = data_cache.get(ticker, pd.DataFrame())
                if df.empty or len(df) < 260: continue

                test_days = df[(df.index >= test_start) & (df.index <= test_end)].index
                for current_date in test_days:
                    
                    spy_past = spy_df[spy_df.index <= current_date]
                    if spy_past.empty or len(spy_past) < 200: continue
                    spy_today = spy_past.iloc[-1]
                    
                    if pd.isna(spy_today["SMA_200"]) or pd.isna(spy_today["SMA_50"]) or pd.isna(spy_today["SMA_21"]): continue
                    
                    is_bull_or_recovering = float(spy_today["Close"]) > float(spy_today["SMA_50"])
                    
                    past_data = df[df.index <= current_date]
                    if len(past_data) < 310: continue

                    today = past_data.iloc[-1]
                    yesterday = past_data.iloc[-2]
                    lookback_data = past_data.iloc[:-1].copy()

                    if not stock_filter_ok(today, cfg): continue
                    
                    vol_ratio = float(today["Volume"]) / max(float(today["Vol_50"]), 1e-9)
                    if vol_ratio < cfg.breakout_volume_ratio: continue

                    day_range = max(float(today["High"]) - float(today["Low"]), 1e-9)
                    close_strength = (float(today["Close"]) - float(today["Low"])) / day_range
                    if close_strength < cfg.min_breakout_close_strength: continue
                    
                    spy_rs = float(spy_today["ROC_65"])
                    stock_rs_absolute = float(today["ROC_65"])
                    stock_rs_relative = stock_rs_absolute - spy_rs
                    
                    if not is_bull_or_recovering:
                        if stock_rs_relative < cfg.bear_market_rs_threshold:
                            continue 
                    else:
                        if stock_rs_relative < cfg.min_rs_65:
                            continue 

                    pattern = check_classical_patterns(lookback_data)
                    if pattern is None: continue

                    pivot = pattern["pivot_price"]
                    prev_close = float(yesterday["Close"])
                    close = float(today["Close"])
                    
                    if not (prev_close <= pivot and close > pivot): continue

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
                        "Volume_Ratio": round(vol_ratio, 2), "RS_65": round(stock_rs_relative, 4), 
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
    trades_by_date_count = {}

    cand_records = candidates.to_dict("records")
    for cand in cand_records:
        entry_date = pd.Timestamp(cand["Entry_Date"])
        ticker = str(cand["Ticker"])

        if trades_by_date_count.get(entry_date, 0) >= cfg.max_new_trades_per_day:
            continue

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

        past_closed = [t for t in accepted if pd.Timestamp(t["Exit_Date"]) < entry_date]
        recent_closed = sorted(past_closed, key=lambda x: pd.Timestamp(x["Exit_Date"]))[-5:]
        
        allowed_max_positions = cfg.max_positions
        if len(recent_closed) >= 3:
            recent_wins = sum(1 for t in recent_closed if float(t["Pct_Change"]) > 0)
            recent_win_rate = recent_wins / len(recent_closed)
            if recent_win_rate < 0.40:
                allowed_max_positions = 2

        if len(active) >= allowed_max_positions: continue

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
        trades_by_date_count[entry_date] = trades_by_date_count.get(entry_date, 0) + 1
        active.append({
            "Ticker": ticker, "Entry_Date": t["Entry_Date"], "Exit_Date": t["Exit_Date"],
            "Entry_Price": entry_price, "Exit_Price": exit_price, "Shares": shares, "Exit_Fee": exit_fee,
            "Risk_Dollars": t["Risk_Dollars"],
        })

    if not accepted: return pd.DataFrame()
    return pd.DataFrame(accepted).sort_values(["Entry_Date", "Exit_Date", "Ticker"]).reset_index(drop=True)

# ==========================================
# 8. Daily Equity Curve (PURE CASH, NO SPY!)
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
    if trades_df.empty:
        return {"Trades": 0, "Wins": 0, "Losses": 0, "Win_Rate_Pct": 0.0, "Avg_Trade_Pct": 0.0, "Avg_Win_Pct": 0.0, "Avg_Loss_Pct": 0.0, "Total_Return_Pct": 0.0, "Max_Drawdown_Pct": 0.0, "Net_PnL": 0.0}

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
    print("VCP BACKTEST REPORT (v46 - Pure Alpha Sniper)")
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

    print("\n--- 📖 CASE STUDIES: עסקאות מופת לפי תבנית ---")
    if not accepted_df.empty:
        for pattern in accepted_df['Pattern_Type'].unique():
            pattern_df = accepted_df[accepted_df['Pattern_Type'] == pattern]
            if pattern_df.empty: continue
            
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
