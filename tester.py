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
    max_alloc_pct: float = 0.12        # הוגדל קצת כדי לנצל את הסבלנות
    max_positions: int = 8
    max_portfolio_heat: float = 0.04
    cooldown_days: int = 15
    slippage_bps: float = 12
    commission_bps: float = 2
    breakout_volume_ratio: float = 1.1 
    min_dollar_vol_50: float = 25_000_000
    min_price: float = 15.0
    
    # --- התאמות לסטופ הדוק ---
    min_risk_pct: float = 0.01         # נאפשר סטופ קרוב מאוד (1%) בלי לפסול את העסקה
    max_risk_pct: float = 0.04         # מקסימום 4% סיכון לעסקה מהכניסה
    
    max_hold_bars: int = 120
    time_stop_bars: int = 30
    min_profit_after_time_stop: float = 0.01
    min_prior_uptrend: float = 0.08
    min_cup_depth: float = 0.04
    max_cup_depth: float = 0.35
    max_handle_depth: float = 0.10
    min_handle_days: int = 3
    max_handle_days: int = 20
    min_cup_days: int = 15
    max_cup_days: int = 200
    max_pivot_extension: float = 0.03
    max_entry_extension: float = 0.03
    max_gap_above_pivot: float = 0.02
    min_breakout_close_strength: float = 0.30
    min_rs_65: float = 0.00
    max_dist_from_52w_high: float = 0.15
    early_exit_bars: int = 10
    early_exit_min_progress: float = -0.02
    min_tight_closes_in_handle: int = 0
    use_point_in_time_universe: bool = False
    raw_price_mode: bool = False
    allow_same_day_cash_reuse: bool = False
    universe_file: str | None = None
    output_prefix: str = "canslim_v9_vcp"

# ==========================================
# 1. Data & Caching
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

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Prev_Close"]).abs(),
        (df["Low"] - df["Prev_Close"]).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"]

    return df

def get_data(ticker: str, start_fetch: str, end_fetch: str, cfg: BacktestConfig, retries: int = 3) -> pd.DataFrame:
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    price_tag = "raw" if cfg.raw_price_mode else "adj"
    cache_file = cache_dir / f"{ticker}_{start_fetch}_{end_fetch}_{price_tag}.pkl"

    if cache_file.exists():
        return pd.read_pickle(cache_file)

    for _ in range(retries):
        try:
            df = yf.Ticker(ticker).history(
                start=start_fetch,
                end=end_fetch,
                auto_adjust=not cfg.raw_price_mode,
                actions=False,
            )
            if not df.empty:
                df = add_indicators(df)
                df.to_pickle(cache_file)
                return df
            return df
        except Exception:
            time.sleep(1.5)

    return pd.DataFrame()

# ==========================================
# 2. Universe membership
# ==========================================
def load_universe_membership(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    u = pd.read_csv(p)
    cols = {c.lower(): c for c in u.columns}
    if "ticker" not in cols:
        raise ValueError("Universe file must include a 'Ticker' column")

    out = u.copy()
    for c in ["start_date", "end_date"]:
        if c in cols:
            out[cols[c]] = pd.to_datetime(out[cols[c]])
    if "year" in cols:
        out[cols["year"]] = out[cols["year"]].astype(int)
    return out

def ticker_allowed_on_date(ticker: str, dt: pd.Timestamp, universe_df: pd.DataFrame | None) -> bool:
    if universe_df is None: return True
    cols = {c.lower(): c for c in universe_df.columns}
    tcol = cols["ticker"]
    sub = universe_df[universe_df[tcol].astype(str).str.upper() == ticker.upper()]
    if sub.empty: return False

    if "year" in cols:
        return bool((sub[cols["year"]] == dt.year).any())
    if "start_date" in cols and "end_date" in cols:
        return bool(((sub[cols["start_date"]] <= dt) & (sub[cols["end_date"]] >= dt)).any())
    return True

def get_sector_for_ticker(ticker: str, dt: pd.Timestamp, universe_df: pd.DataFrame | None) -> str:
    if universe_df is None: return "UNKNOWN"
    cols = {c.lower(): c for c in universe_df.columns}
    if "sector" not in cols: return "UNKNOWN"
    tcol = cols["ticker"]
    sub = universe_df[universe_df[tcol].astype(str).str.upper() == ticker.upper()].copy()
    if sub.empty: return "UNKNOWN"
    if "year" in cols:
        sub = sub[sub[cols["year"]] == dt.year]
    elif "start_date" in cols and "end_date" in cols:
        sub = sub[(sub[cols["start_date"]] <= dt) & (sub[cols["end_date"]] >= dt)]
    if sub.empty: return "UNKNOWN"
    return str(sub.iloc[0][cols["sector"]])

# ==========================================
# 3. Filters
# ==========================================
def market_filter_ok(spy_df: pd.DataFrame, current_date: pd.Timestamp) -> bool:
    x = spy_df[spy_df.index <= current_date]
    if len(x) < 220: return False

    row = x.iloc[-1]
    sma200_old = x["SMA_200"].iloc[-20]

    # תיקון השגיאות של הפנדס - any() כדי למנוע ambiguity 
    if any(pd.isna(row[c]).any() if isinstance(row[c], pd.Series) else pd.isna(row[c]) for c in ["SMA_50", "SMA_150", "SMA_200", "ROC_20", "ROC_65"]):
        return False
    if pd.isna(sma200_old).any() if isinstance(sma200_old, pd.Series) else pd.isna(sma200_old):
        return False

    return (
        float(row["Close"]) > float(row["SMA_50"]) > float(row["SMA_150"]) > float(row["SMA_200"]) and 
        float(row["SMA_200"]) > float(sma200_old) and
        float(row["ROC_20"]) > -0.03 and 
        float(row["ROC_65"]) > 0
    )

def stock_filter_ok(today: pd.Series, cfg: BacktestConfig) -> bool:
    required = ["SMA_21", "SMA_50", "SMA_150", "SMA_200", "Vol_50", "ATR_14", "ATR_Pct", "ROC_65", "DollarVol_50", "High_252"]
    
    for c in required:
        if pd.isna(today[c]).any() if isinstance(today[c], pd.Series) else pd.isna(today[c]):
            return False

    if float(today["Close"]) < cfg.min_price: return False
    if not (float(today["Close"]) > float(today["SMA_21"]) > float(today["SMA_50"]) > float(today["SMA_150"]) > float(today["SMA_200"])): return False
    if float(today["ROC_65"]) < cfg.min_prior_uptrend: return False
    if float(today["DollarVol_50"]) < cfg.min_dollar_vol_50: return False
    if not (0.01 <= float(today["ATR_Pct"]) <= 0.08): return False

    dist_52w = (float(today["Close"]) / float(today["High_252"])) - 1.0
    if dist_52w < -cfg.max_dist_from_52w_high: return False

    return True

# ==========================================
# 4. Pattern detection
# ==========================================
def today_close_distance_to_pivot(close_price: float, pivot: float) -> float:
    if pivot <= 0: return 0.0
    d = abs(close_price / pivot - 1.0)
    return max(0.0, 0.05 - d)

def handle_tightness_score(handle: pd.DataFrame) -> float:
    if len(handle) < 3: return 0.0
    tight = 0
    for i in range(1, len(handle)):
        prev_c = float(handle["Close"].iloc[i - 1])
        cur_c = float(handle["Close"].iloc[i])
        if prev_c > 0 and abs(cur_c / prev_c - 1.0) <= 0.012:
            tight += 1
    return tight / max(len(handle) - 1, 1)

def cup_roundness_ok(recent: pd.DataFrame, left_pos: int, bottom_pos: int, right_pos: int) -> bool:
    left_leg = recent.iloc[left_pos:bottom_pos + 1]
    right_leg = recent.iloc[bottom_pos:right_pos + 1]
    if len(left_leg) < 8 or len(right_leg) < 8: return False
    left_days = bottom_pos - left_pos
    right_days = right_pos - bottom_pos
    if left_days <= 0 or right_days <= 0: return False
    symmetry = min(left_days, right_days) / max(left_days, right_days)
    if symmetry < 0.35: return False

    bottom_zone = recent.iloc[max(0, bottom_pos - 5): bottom_pos + 6]
    if len(bottom_zone) < 5: return False
    bottom_range = (bottom_zone["High"].max() - bottom_zone["Low"].min()) / max(float(bottom_zone["Low"].min()), 1e-9)
    if bottom_range > 0.15: return False
    return True

def get_cup_handle_signal(pattern_data: pd.DataFrame, cfg: BacktestConfig):
    recent = pattern_data.tail(250).copy()
    if len(recent) < 250: return None

    smooth_high = recent["High"].rolling(window=3, min_periods=1, center=True).mean()
    smooth_low = recent["Low"].rolling(window=3, min_periods=1, center=True).mean()

    search_cutoff = max(cfg.min_handle_days + 1, 8)
    left_peak_idx = smooth_high.iloc[:-search_cutoff].idxmax()
    left_pos = recent.index.get_loc(left_peak_idx)
    left_peak = float(recent.loc[left_peak_idx, "High"])

    if left_pos < 20 or left_pos > 160: return None

    prior_slice = recent.iloc[max(0, left_pos - 65):left_pos]
    if len(prior_slice) < 40: return None
    prior_uptrend = (left_peak / float(prior_slice["Close"].iloc[0])) - 1.0
    if prior_uptrend < cfg.min_prior_uptrend: return None

    bottom_slice = smooth_low.iloc[left_pos + 5: -cfg.min_handle_days]
    if len(bottom_slice) < 20: return None
    bottom_idx = bottom_slice.idxmin()
    bottom_pos = recent.index.get_loc(bottom_idx)
    cup_low = float(recent.loc[bottom_idx, "Low"])

    rs_end = -cfg.min_handle_days + 1 if cfg.min_handle_days > 1 else None
    right_slice = smooth_high.iloc[bottom_pos + 10: rs_end]
    if len(right_slice) < 15: return None
    right_peak_idx = right_slice.idxmax()
    right_pos = recent.index.get_loc(right_peak_idx)
    right_peak = float(recent.loc[right_peak_idx, "High"])

    cup_len = right_pos - left_pos
    if not (cfg.min_cup_days <= cup_len <= cfg.max_cup_days): return None
    if not cup_roundness_ok(recent, left_pos, bottom_pos, right_pos): return None

    rim_price = min(left_peak, right_peak)
    cup_depth_pct = (rim_price - cup_low) / rim_price
    if not (cfg.min_cup_depth <= cup_depth_pct <= cfg.max_cup_depth): return None
    if right_peak < left_peak * 0.93: return None

    handle = recent.iloc[right_pos + 1:].copy()
    handle_len = len(handle)
    if not (cfg.min_handle_days <= handle_len <= cfg.max_handle_days): return None

    handle_low = float(handle["Low"].min())
    
    # --- שילוב המשולש השורי / VCP ---
    # מוצא את אזור הכיווץ הממש אחרון (5 ימים אחרונים או פחות אם הידית קצרה)
    last_tight_days = handle.tail(5) if len(handle) >= 5 else handle
    tight_low = float(last_tight_days["Low"].min())
    
    # בדיקת שפל עולה ברור מול תחתית הספל
    if tight_low <= cup_low * 1.02: 
        return None

    handle_depth_pct = (rim_price - handle_low) / rim_price
    if not (0 < handle_depth_pct <= cfg.max_handle_depth): return None

    cup_mid = cup_low + 0.5 * (rim_price - cup_low)
    if handle_low < cup_mid: return None

    handle_vol = handle["Volume"].mean()
    pre_handle_vol = recent.iloc[max(0, right_pos - 20): right_pos]["Volume"].mean()
    if pd.isna(handle_vol) or pd.isna(pre_handle_vol) or handle_vol >= pre_handle_vol * 0.90: return None

    tightness = handle_tightness_score(handle)
    tight_days = 0
    for i in range(1, len(handle)):
        prev_c = float(handle["Close"].iloc[i - 1])
        cur_c = float(handle["Close"].iloc[i])
        if prev_c > 0 and abs(cur_c / prev_c - 1.0) <= 0.012:
            tight_days += 1

    if tight_days < cfg.min_tight_closes_in_handle: return None

    pivot = float(handle["High"].max())

    score = (
        today_close_distance_to_pivot(float(recent["Close"].iloc[-1]), pivot) +
        min(prior_uptrend, 0.60) +
        max(0.0, 0.18 - handle_depth_pct) +
        tightness * 0.15
    )

    return {
        "pivot_price": pivot,
        "handle_low": handle_low,
        "tight_low": tight_low,  # הוספתי למילון כדי להשתמש בזה לסטופ ההדוק
        "cup_depth_pct": cup_depth_pct,
        "handle_depth_pct": handle_depth_pct,
        "cup_length": cup_len,
        "handle_length": handle_len,
        "prior_uptrend": prior_uptrend,
        "tightness": round(tightness, 4),
        "score": round(score, 4),
    }

def compute_signal_score(pattern_score: float, rs_65: float, volume_ratio: float, dist_52w: float, close_strength: float) -> float:
    rs_component = max(min(rs_65, 0.35), -0.05)
    vol_component = max(0.0, min(volume_ratio - 1.0, 2.0)) * 0.12
    near_high_component = max(0.0, 0.10 - abs(dist_52w))
    cs_component = max(0.0, close_strength - 0.5) * 0.20
    return round(pattern_score + rs_component + vol_component + near_high_component + cs_component, 4)

# ==========================================
# 5. Trade simulation
# ==========================================
def classify_pnl(pct: float) -> str:
    if pct > 0: return "Win"
    if pct < 0: return "Loss"
    return "Flat"

def simulate_trade(df: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float, initial_stop: float, initial_atr: float, pivot: float, cfg: BacktestConfig):
    future = df[df.index >= entry_date].head(cfg.max_hold_bars)
    if future.empty: return None

    stop_today = initial_stop
    stop_next_day = initial_stop
    highest_seen = entry_price
    lowest_seen = entry_price
    mfe_pct = 0.0
    mae_pct = 0.0

    final_exit_date = future.index[-1]
    final_exit_price = float(future.iloc[-1]["Close"]) * (1 - cfg.slippage_bps / 10000)
    exit_reason = "MaxHold"

    for i, row in enumerate(future.itertuples()):
        dt = row.Index
        day_open = float(row.Open)
        day_high = float(row.High)
        day_low = float(row.Low)
        day_close = float(row.Close)

        stop_today = initial_stop if i == 0 else stop_next_day
        raw_atr = getattr(row, "ATR_14", np.nan)
        atr_live = float(raw_atr) if pd.notna(raw_atr) and float(raw_atr) > 0 else initial_atr

        if day_open <= stop_today:
            final_exit_date = dt
            final_exit_price = day_open * (1 - cfg.slippage_bps / 10000)
            exit_reason = "GapStop"
            lowest_seen = min(lowest_seen, day_open)
            mae_pct = min(mae_pct, (lowest_seen - entry_price) / entry_price * 100)
            break

        if day_low <= stop_today:
            final_exit_date = dt
            final_exit_price = stop_today * (1 - cfg.slippage_bps / 10000)
            exit_reason = "StopHit"
            lowest_seen = min(lowest_seen, day_low)
            mae_pct = min(mae_pct, (lowest_seen - entry_price) / entry_price * 100)
            break

        highest_seen = max(highest_seen, day_high)
        lowest_seen = min(lowest_seen, day_low)
        mfe_pct = max(mfe_pct, (highest_seen - entry_price) / entry_price * 100)
        mae_pct = min(mae_pct, (lowest_seen - entry_price) / entry_price * 100)

        if (i + 1) == cfg.early_exit_bars:
            progress = (day_close / entry_price) - 1.0
            # הסרתי את החיתוך המוקדם אם day_close < pivot, כי הסטופ לוס ההדוק החדש כבר עושה את העבודה טוב יותר
            if progress < cfg.early_exit_min_progress:
                final_exit_date = dt
                final_exit_price = day_close * (1 - cfg.slippage_bps / 10000)
                exit_reason = "EarlyFail"
                break

        profit_high = (highest_seen - entry_price) / entry_price
        new_stop = stop_today
        
        # --- ניהול הסטופ הסבלני ---
        # ברייק-איוון רק אחרי שהפריצה מוכיחה את עצמה ממש
        if profit_high >= 0.06: new_stop = max(new_stop, entry_price * 1.005)
        if profit_high >= 0.15: new_stop = max(new_stop, highest_seen * 0.90)
        if profit_high >= 0.25: new_stop = max(new_stop, highest_seen * 0.88)

        stop_next_day = max(stop_today, new_stop)

        if (i + 1) >= cfg.time_stop_bars:
            if (day_close - entry_price) / entry_price < cfg.min_profit_after_time_stop:
                final_exit_date = dt
                final_exit_price = day_close * (1 - cfg.slippage_bps / 10000)
                exit_reason = "TimeExit"
                break

    gross_pct = (final_exit_price - entry_price) / entry_price * 100
    net_pct = gross_pct - (2 * cfg.commission_bps / 100)
    risk_per_share = max(entry_price - initial_stop, 1e-9)
    r_multiple = (entry_price * net_pct / 100) / risk_per_share

    return {
        "Exit_Date": final_exit_date,
        "Exit_Price": final_exit_price,
        "Exit_Reason": exit_reason,
        "Pct_Change": round(net_pct, 2),
        "MFE_Pct": round(mfe_pct, 2),
        "MAE_Pct": round(mae_pct, 2),
        "R_Multiple": round(r_multiple, 2),
        "Hold_Bars": len(future[future.index <= final_exit_date]),
    }

# ==========================================
# 6. Candidate generation
# ==========================================
def generate_candidate_trades(tickers, data_cache, spy_df, cfg: BacktestConfig, universe_df=None):
    candidates = []
    print("\nScanning for signals...")

    for year in tqdm(range(cfg.start_year, cfg.end_year + 1), desc="Years"):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        for ticker in tickers:
            df = data_cache.get(ticker, pd.DataFrame())
            if df.empty or len(df) < 260: continue

            test_days = df[(df.index >= test_start) & (df.index <= test_end)].index
            for current_date in test_days:
                if cfg.use_point_in_time_universe and not ticker_allowed_on_date(ticker, current_date, universe_df): continue
                if not market_filter_ok(spy_df, current_date): continue

                past_data = df[df.index <= current_date]
                if len(past_data) < 251: continue

                pattern_data = past_data.iloc[:-1].copy()
                today = past_data.iloc[-1]

                if len(pattern_data) < 250: continue
                if not stock_filter_ok(today, cfg): continue

                pattern = get_cup_handle_signal(pattern_data, cfg)
                if pattern is None: continue

                pivot = pattern["pivot_price"]
                prev_close = float(pattern_data.iloc[-1]["Close"])
                close = float(today["Close"])
                high_today = float(today["High"])
                low_today = float(today["Low"])

                day_range = max(high_today - low_today, 1e-9)
                close_strength = (close - low_today) / day_range
                if close_strength < cfg.min_breakout_close_strength: continue

                vol_ratio = float(today["Volume"]) / float(today["Vol_50"]) if today["Vol_50"] > 0 else np.nan
                if pd.isna(vol_ratio) or vol_ratio < cfg.breakout_volume_ratio: continue

                if not (prev_close <= pivot and high_today > pivot and close > pivot and close <= pivot * (1 + cfg.max_pivot_extension)):
                    continue

                spy_row = spy_df[spy_df.index <= current_date]
                if spy_row.empty or pd.isna(spy_row.iloc[-1].get("ROC_65", np.nan)): continue
                rs_65 = float(today["ROC_65"]) - float(spy_row.iloc[-1]["ROC_65"])
                if rs_65 < cfg.min_rs_65: continue

                high_252 = float(today["High_252"])
                if high_252 <= 0 or pd.isna(high_252): continue
                dist_52w = (close / high_252) - 1.0
                if dist_52w < -cfg.max_dist_from_52w_high: continue

                next_bar = df[df.index > current_date].head(1)
                if next_bar.empty: continue

                entry_date = next_bar.index[0]
                entry_open = float(next_bar.iloc[0]["Open"])

                gap_from_pivot = (entry_open / pivot) - 1.0
                if gap_from_pivot > cfg.max_gap_above_pivot: continue

                entry_price = entry_open * (1 + cfg.slippage_bps / 10000)
                if entry_price > pivot * (1 + cfg.max_entry_extension): continue

                atr = float(today["ATR_14"])
                if np.isnan(atr) or atr <= 0: continue

                # --- הסטופ לוס ההדוק ---
                # הסטופ מונח קצת מתחת לנמוך של כיווץ הימים האחרונים (tight_low)
                tight_low = float(pattern["tight_low"])
                calculated_stop = tight_low * 0.99
                # הגנה: הסטופ לא יכול להיות עמוק יותר מהמקסימום סיכון המותר (4%)
                max_allowed_stop = entry_price * (1 - cfg.max_risk_pct)
                
                initial_stop = max(calculated_stop, max_allowed_stop)
                
                risk_pct = (entry_price - initial_stop) / entry_price
                if not (cfg.min_risk_pct <= risk_pct <= cfg.max_risk_pct): continue

                sim = simulate_trade(df, entry_date, entry_price, initial_stop, atr, pivot, cfg)
                if sim is None: continue

                total_score = compute_signal_score(pattern["score"], rs_65, vol_ratio, dist_52w, close_strength)

                candidates.append({
                    "Year": year, "Ticker": ticker, "Sector": get_sector_for_ticker(ticker, current_date, universe_df),
                    "Signal_Date": current_date, "Entry_Date": entry_date, "Entry_Price": round(entry_price, 2),
                    "Exit_Date": sim["Exit_Date"], "Exit_Price": round(sim["Exit_Price"], 2), "Pct_Change": sim["Pct_Change"],
                    "Risk_Pct": round(risk_pct * 100, 2), "Stop_Price": round(initial_stop, 2),
                    "Cup_Depth_Pct": round(pattern["cup_depth_pct"] * 100, 2), "Handle_Depth_Pct": round(pattern["handle_depth_pct"] * 100, 2),
                    "Cup_Length": int(pattern["cup_length"]), "Handle_Length": int(pattern["handle_length"]),
                    "Prior_Uptrend_Pct": round(pattern["prior_uptrend"] * 100, 2), "Handle_Tightness": pattern["tightness"],
                    "Volume_Ratio": round(vol_ratio, 2), "RS_65": round(rs_65, 4), "Dist_52W_High": round(dist_52w, 4),
                    "Close_Strength": round(close_strength, 4), "Gap_From_Pivot": round(gap_from_pivot, 4),
                    "Hold_Bars": sim["Hold_Bars"], "Result": classify_pnl(sim["Pct_Change"]),
                    "Exit_Reason": sim["Exit_Reason"], "MFE_Pct": sim["MFE_Pct"], "MAE_Pct": sim["MAE_Pct"],
                    "R_Multiple": sim["R_Multiple"], "Pattern_Score": pattern["score"], "Signal_Score": total_score,
                })

    if not candidates: return pd.DataFrame()
    return (pd.DataFrame(candidates)
            .sort_values(["Entry_Date", "Signal_Score", "Volume_Ratio", "RS_65"], ascending=[True, False, False, False])
            .reset_index(drop=True))

# ==========================================
# 7. Portfolio acceptance
# ==========================================
def get_close_on_or_before(df: pd.DataFrame, dt: pd.Timestamp, fallback: float) -> float:
    x = df[df.index <= dt]
    return float(x.iloc[-1]["Close"]) if not x.empty else fallback

def current_portfolio_heat(active: list[dict]) -> float:
    return sum(float(pos.get("Risk_Dollars", 0.0)) for pos in active)

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
        sector = str(cand.get("Sector", "UNKNOWN"))

        if ticker in last_exit_by_ticker and entry_date <= last_exit_by_ticker[ticker] + pd.Timedelta(days=cfg.cooldown_days):
            continue

        release, still_active = [], []
        for pos in active:
            exit_dt = pd.Timestamp(pos["Exit_Date"])
            closed = (exit_dt < entry_date or (cfg.allow_same_day_cash_reuse and exit_dt == entry_date))
            (release if closed else still_active).append(pos)

        for pos in release:
            cash += pos["Shares"] * pos["Exit_Price"] - pos["Exit_Fee"]
        active = still_active

        if any(pos["Ticker"] == ticker for pos in active): continue
        if len(active) >= cfg.max_positions: continue
        if sector != "UNKNOWN" and sum(1 for pos in active if pos.get("Sector") == sector) >= 2: continue

        equity = cash + sum(
            get_close_on_or_before(data_cache[p["Ticker"]], entry_date, p["Entry_Price"]) * p["Shares"]
            for p in active
        )

        entry_price = float(cand["Entry_Price"])
        stop_price = float(cand["Stop_Price"])
        exit_price = float(cand["Exit_Price"])

        risk_per_share = max(entry_price - stop_price, 1e-9)
        max_risk_dollars_trade = equity * cfg.risk_per_trade
        current_heat = current_portfolio_heat(active)
        max_heat = equity * cfg.max_portfolio_heat
        remaining_heat = max(0.0, max_heat - current_heat)

        if remaining_heat <= 0: continue

        shares_by_risk = min(max_risk_dollars_trade, remaining_heat) / risk_per_share
        shares_by_alloc = (equity * cfg.max_alloc_pct) / entry_price
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
        t["Heat_Pct"] = round(risk_dollars / equity * 100, 2) if equity > 0 else 0.0

        accepted.append(t)
        last_exit_by_ticker[ticker] = pd.Timestamp(t["Exit_Date"])
        active.append({
            "Ticker": ticker, "Sector": sector, "Entry_Date": t["Entry_Date"], "Exit_Date": t["Exit_Date"],
            "Entry_Price": entry_price, "Exit_Price": exit_price, "Shares": shares, "Exit_Fee": exit_fee,
            "Risk_Dollars": risk_dollars,
        })

    if not accepted: return pd.DataFrame()
    return pd.DataFrame(accepted).sort_values(["Entry_Date", "Exit_Date", "Ticker"]).reset_index(drop=True)

# ==========================================
# 8. Daily equity curve
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
            "Equity": round(equity, 2), "Drawdown_Pct": round(dd, 2), "Open_Positions": len(open_pos),
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
        "Trades": 0, "Wins": 0, "Losses": 0, "Flats": 0, "TimeExits": 0,
        "Win_Rate_Pct": 0.0, "Avg_Trade_Pct": 0.0, "Avg_Win_Pct": 0.0, "Avg_Loss_Pct": 0.0,
        "Profit_Factor": 0.0, "Expectancy_Pct": 0.0, "Avg_R": 0.0,
        "Avg_MFE_Pct": 0.0, "Avg_MAE_Pct": 0.0, "Avg_Hold_Bars": 0.0,
        "Total_Return_Pct": 0.0, "Max_Drawdown_Pct": 0.0, "Net_PnL": 0.0,
    }
    if trades_df.empty: return empty

    wins = trades_df[trades_df["Pct_Change"] > 0]
    losses = trades_df[trades_df["Pct_Change"] < 0]
    flats = trades_df[trades_df["Pct_Change"] == 0]

    gp = wins["Pct_Change"].sum()
    gl = abs(losses["Pct_Change"].sum())
    pf = round(gp / gl, 2) if gl > 0 else np.nan

    total_return, max_dd = 0.0, 0.0
    if equity_df is not None and not equity_df.empty:
        total_return = round((equity_df["Equity"].iloc[-1] / equity_df["Equity"].iloc[0] - 1.0) * 100, 2)
        max_dd = calc_drawdown(equity_df["Equity"])

    return {
        "Trades": len(trades_df), "Wins": len(wins), "Losses": len(losses), "Flats": len(flats),
        "TimeExits": int((trades_df["Exit_Reason"] == "TimeExit").sum()),
        "Win_Rate_Pct": round(len(wins) / len(trades_df) * 100, 2),
        "Avg_Trade_Pct": round(trades_df["Pct_Change"].mean(), 2),
        "Avg_Win_Pct": round(wins["Pct_Change"].mean(), 2) if len(wins) else 0.0,
        "Avg_Loss_Pct": round(losses["Pct_Change"].mean(), 2) if len(losses) else 0.0,
        "Profit_Factor": pf, "Expectancy_Pct": round(trades_df["Pct_Change"].mean(), 2),
        "Avg_R": round(trades_df["R_Multiple"].mean(), 2),
        "Avg_MFE_Pct": round(trades_df["MFE_Pct"].mean(), 2),
        "Avg_MAE_Pct": round(trades_df["MAE_Pct"].mean(), 2),
        "Avg_Hold_Bars": round(trades_df["Hold_Bars"].mean(), 2),
        "Total_Return_Pct": total_return, "Max_Drawdown_Pct": max_dd,
        "Net_PnL": round(trades_df["Net_PnL"].sum(), 2) if "Net_PnL" in trades_df.columns else 0.0,
    }

def yearly_summary(accepted_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    if accepted_df.empty: return pd.DataFrame()
    tmp = accepted_df.copy()
    tmp["Entry_Date"] = pd.to_datetime(tmp["Entry_Date"])
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
            Win_Rate_Pct=("Pct_Change", lambda s: round(s.gt(0).sum() / len(s) * 100, 2)),
            Avg_Trade_Pct=("Pct_Change", lambda s: round(s.mean(), 2)),
            Net_PnL=("Net_PnL", "sum"),
        )
        .reset_index()
    )

# ==========================================
# 10. Orchestrator
# ==========================================
def run_canslim_backtest_v7(tickers: list[str], cfg: BacktestConfig):
    tickers = sorted({t.upper() for t in tickers})
    universe_df = load_universe_membership(cfg.universe_file)
    start_fetch = f"{cfg.start_year - 1}-01-01"
    end_fetch = f"{cfg.end_year + 1}-02-01"

    print(f"\nCANSLIM Patient VCP Backtest: {cfg.start_year}-{cfg.end_year}")
    print(f"raw_price_mode={cfg.raw_price_mode} | time_stop_bars={cfg.time_stop_bars} | min_dollar_vol=${cfg.min_dollar_vol_50:,}")

    if cfg.use_point_in_time_universe and cfg.universe_file is None:
        raise ValueError("For serious backtests, provide universe_file with point-in-time membership.")
    print("-" * 80)

    spy = get_data(cfg.benchmark, start_fetch, end_fetch, cfg)
    if spy.empty or len(spy) < 220: raise ValueError("Could not load benchmark data")
    data_cache = {cfg.benchmark: spy}

    print(f"Downloading/Loading {len(tickers)} tickers...")
    for ticker in tqdm(tickers, desc="Tickers"):
        data_cache[ticker] = get_data(ticker, start_fetch, end_fetch, cfg)

    candidates_df = generate_candidate_trades(tickers, data_cache, spy, cfg, universe_df=universe_df)
    accepted_df = accept_trades_with_portfolio_rules(candidates_df, data_cache, cfg)

    if not accepted_df.empty:
        accepted_df = accepted_df.sort_values(["Entry_Date", "Exit_Date", "Ticker"]).reset_index(drop=True)

    equity_df = build_daily_equity_curve(accepted_df, data_cache, spy, cfg)
    yearly_df = yearly_summary(accepted_df, equity_df)
    monthly_df = monthly_summary(accepted_df)
    overall = summarize_trades(accepted_df, equity_df)

    return candidates_df, accepted_df, equity_df, yearly_df, monthly_df, overall

# ==========================================
# 11. Output helpers
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

def print_report(overall: dict, yearly_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("CANSLIM v9 (Patient & Tight VCP) - Backtest Report")
    print("=" * 80)
    for _, r in yearly_df.iterrows():
        print(f" {int(r['Year'])}: trades={int(r['Trades']):3d} | WR={r['Win_Rate_Pct']:5.1f}% | avgTrade={r['Avg_Trade_Pct']:+5.2f}% | ret={r['Total_Return_Pct']:+6.2f}% | MDD={r['Max_Drawdown_Pct']:5.2f}%")
    print("-" * 80)
    print(f" Total Trades  : {overall['Trades']}")
    print(f" Win Rate      : {overall['Win_Rate_Pct']}%")
    print(f" Avg Trade     : {overall['Avg_Trade_Pct']}%")
    print(f" Profit Factor : {overall['Profit_Factor']}")
    print(f" Total Return  : {overall['Total_Return_Pct']}%")
    print(f" Max Drawdown  : {overall['Max_Drawdown_Pct']}%")
    print(f" Net PnL       : ${overall['Net_PnL']:,.0f}")
    print("=" * 80)

# ==========================================
# 12. Utilities
# ==========================================
def fetch_sp500_tickers_current() -> list[str]:
    # --- תיקון שגיאת 403 של ויקיפדיה ---
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            tables = pd.read_html(response.read())
        tickers = tables[0]["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)
    except Exception:
        return example_tickers()

def example_tickers() -> list[str]:
    return sorted({
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "TSM", "ASML", "LRCX",
        "INTU", "CRM", "PANW", "NOW", "LLY", "ISRG", "COST", "UNH", "SPGI", "NOC",
    })

# ==========================================
# 13. Main
# ==========================================
if __name__ == "__main__":
    cfg = BacktestConfig()

    tickers = fetch_sp500_tickers_current() if not cfg.use_point_in_time_universe else sorted(pd.read_csv(cfg.universe_file)["Ticker"].astype(str).str.upper().unique())

    candidates_df, accepted_df, equity_df, yearly_df, monthly_df, overall = run_canslim_backtest_v7(tickers, cfg)

    print_report(overall, yearly_df)
    save_outputs(candidates_df, accepted_df, equity_df, yearly_df, monthly_df, overall, cfg)
