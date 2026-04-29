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
    
    # פרמטרים גיאומטריים - משולש עולה (Ascending Triangle)
    min_prior_uptrend: float = 0.08    
    max_base_depth: float = 0.35       
    max_tightness_depth: float = 0.08  
    min_breakout_close_strength: float = 0.30
    max_pivot_extension: float = 0.04  
    max_dist_from_52w_high: float = 0.15
    min_price: float = 15.0
    min_dollar_vol_50: float = 25_000_000
    
    # ניהול טרייד
    min_risk_pct: float = 0.01         
    max_risk_pct: float = 0.045        
    early_exit_bars: int = 10          
    early_exit_min_progress: float = -0.02 
    time_stop_bars: int = 35           
    min_profit_after_time_stop: float = 0.01 
    max_hold_bars: int = 120           
    
    use_point_in_time_universe: bool = False
    raw_price_mode: bool = False
    allow_same_day_cash_reuse: bool = False
    universe_file: str | None = None
    output_prefix: str = "ascending_triangle_v10"

# ==========================================
# 1. Data & Indicators
# ==========================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    
    # הסרת כפילויות במקרה של נתוני יאהו משובשים
    df = df[~df.index.duplicated(keep='first')]

    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_150"] = df["Close"].rolling(150).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Vol_50"] = df["Volume"].rolling(50).mean()
    df["DollarVol_50"] = df["Close"].rolling(50).mean() * df["Volume"].rolling(50).mean()
    df["Prev_Close"] = df["Close"].shift(1)
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
# 2. Universe Handling
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

    # שימוש במערכי numpy כדי למנוע קריסה מאינדקסים כפולים (Duplicate DatetimeIndex)
    highs = recent["High"].values
    lows = recent["Low"].values

    # --- שלב 1: מציאת השיא המרכזי ---
    slice1 = highs[:-10]
    if len(slice1) == 0: return None
    peak1_pos = int(np.argmax(slice1))
    peak1_price = float(slice1[peak1_pos])

    if peak1_pos < 30: return None 

    # --- שלב 2: העמק הראשון (הנסיגה הגדולה) ---
    post_peak1_lows = lows[peak1_pos+1 : -5]
    if len(post_peak1_lows) < 10: return None

    valley1_pos_rel = int(np.argmin(post_peak1_lows))
    valley1_price = float(post_peak1_lows[valley1_pos_rel])
    valley1_pos = peak1_pos + 1 + valley1_pos_rel

    base_depth = (peak1_price - valley1_price) / peak1_price
    if base_depth > cfg.max_base_depth: return None

    # --- שלב 3: בדיקת התנגדות (פסגה שניה) ---
    post_valley1_highs = highs[valley1_pos+1 : -2]
    if len(post_valley1_highs) < 5: return None

    peak2_pos_rel = int(np.argmax(post_valley1_highs))
    peak2_price = float(post_valley1_highs[peak2_pos_rel])
    peak2_pos = valley1_pos + 1 + peak2_pos_rel

    if peak2_price < peak1_price * 0.96 or peak2_price > peak1_price * 1.02:
        return None

    # --- שלב 4: העמק השני (שפלים עולים וכיווץ) ---
    post_peak2_lows = lows[peak2_pos+1 :]
    if len(post_peak2_lows) < 3: return None

    valley2_price = float(np.min(post_peak2_lows))
    
    # חוק השפלים העולים
    if valley2_price <= valley1_price * 1.02: 
        return None

    # חוק ההידוק: כיווץ סופי לפני הפריצה
    tightness = (peak2_price - valley2_price) / peak2_price
    if tightness > cfg.max_tightness_depth: 
        return None

    pivot = max(peak1_price, peak2_price)

    return {
        "pivot_price": pivot, 
        "tight_low": valley2_price 
    }

# ==========================================
# 5. Patient Trade Simulation (TIGHT STOP)
# ==========================================
def simulate_trade(df: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float, tight_low: float, cfg: BacktestConfig):
    future = df[df.index >= entry_date].head(cfg.max_hold_bars)
    if future.empty: return None
    
    calculated_stop = tight_low * 0.99
    max_loss_stop = entry_price * (1 - cfg.max_risk_pct)
    stop_price = max(calculated_stop, max_loss_stop)
    
    if stop_price >= entry_price:
        stop_price = entry_price * 0.985 
        
    highest_seen = float(entry_price)

    for i, row in enumerate(future.itertuples()):
        day_low = float(row.Low)
        day_high = float(row.High)
        day_close = float(row.Close)

        highest_seen = max(highest_seen, day_high)
        profit = (highest_seen / entry_price) - 1

        if profit >= 0.06: stop_price = max(stop_price, entry_price * 1.005)
        if profit >= 0.15: stop_price = max(stop_price, highest_seen * 0.90)
        if profit >= 0.25: stop_price = max(stop_price, highest_seen * 0.88)

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
    for ticker in tqdm(tickers, desc="Scanning Geometry"):
        df = data_cache.get(ticker, pd.DataFrame())
        if df.empty or len(df) < 250: continue
        for i in range(250, len(df)):
            past = df.iloc[:i]
            today = past.iloc[-1]
            if today.name.year < cfg.start_year or not market_filter_ok(spy_df, today.name): continue
            if not stock_filter_ok(today, cfg): continue
            
            pattern = get_ascending_triangle_signal(past, cfg)
            if pattern:
                curr_close = float(today.Close)
                prev_close = float(past.iloc[-2].Close)
                pivot = pattern['pivot_price']
                
                if curr_close > pivot and prev_close <= pivot:
                    if curr_close <= pivot * (1+cfg.max_pivot_extension):
                        if i+1 >= len(df): continue
                        entry_p = float(df.iloc[i+1].Open) * (1+cfg.slippage_bps/10000)
                        
                        sim = simulate_trade(df, df.index[i+1], entry_p, pattern['tight_low'], cfg)
                        if sim:
                            candidates.append({
                                "Year": today.name.year, "Ticker": ticker, "Entry_Date": df.index[i+1], 
                                "Exit_Date": sim["Exit_Date"], "Entry_Price": entry_p, "Exit_Price": sim["Exit_Price"], 
                                "Pct": (sim["Exit_Price"]/entry_p-1)*100, "Reason": sim["Exit_Reason"]
                            })
    if not candidates: return pd.DataFrame()
    return pd.DataFrame(candidates).sort_values("Entry_Date").reset_index(drop=True)

# ==========================================
# 7. Portfolio Management (Real Money Simulation)
# ==========================================
def accept_trades(candidates, data_cache, cfg):
    if candidates.empty: return pd.DataFrame()
    cash, active, accepted = cfg.initial_capital, [], []
    
    for cand in candidates.to_dict("records"):
        dt = pd.Timestamp(cand["Entry_Date"])
        
        for p in active[:]:
            if pd.Timestamp(p["Exit_Date"]) < dt:
                cash += p["Shares"] * p["Exit_Price"] * (1 - cfg.commission_bps/10000)
                active.remove(p)
                
        if len(active) >= cfg.max_positions: continue
        
        current_mkt_val = 0
        for p in active:
            ticker_df = data_cache[p["Ticker"]]
            try:
                curr_p = float(ticker_df.loc[ticker_df.index <= dt].iloc[-1].Close)
            except:
                curr_p = p["Entry_Price"]
            current_mkt_val += curr_p * p["Shares"]
            
        equity = cash + current_mkt_val
        shares = int(min(equity * cfg.max_alloc_pct, cash) / cand["Entry_Price"])
        
        if shares > 0:
            cash -= shares * cand["Entry_Price"] * (1 + cfg.commission_bps/10000)
            cand["Shares"] = shares
            cand["Net_PnL"] = (cand["Exit_Price"] - cand["Entry_Price"]) * shares
            accepted.append(cand)
            active.append(cand)
            
    return pd.DataFrame(accepted)

# ==========================================
# 8. Daily Equity Curve
# ==========================================
def get_close_on_or_before(df, dt, fallback):
    x = df[df.index <= dt]
    return float(x.iloc[-1].Close) if not x.empty else fallback

def build_daily_equity_curve(accepted, data_cache, spy, cfg):
    if accepted.empty: return pd.DataFrame()
    start_dt, end_dt = pd.Timestamp(f"{cfg.start_year}-01-01"), pd.Timestamp(f"{cfg.end_year}-12-31")
    trade_dates = [pd.Timestamp(r["Entry_Date"]) for r in accepted.to_dict("records")] + [pd.Timestamp(r["Exit_Date"]) for r in accepted.to_dict("records")]
    calendar = spy.index.union(pd.DatetimeIndex(trade_dates)).drop_duplicates().sort_values()
    calendar = calendar[(calendar >= start_dt) & (calendar <= end_dt)]
    
    cash, open_pos, rows, peak = cfg.initial_capital, {}, [], cfg.initial_capital
    entries_by_date = accepted.groupby("Entry_Date").apply(lambda x: x.to_dict("records")).to_dict()
    exits_by_date = accepted.groupby("Exit_Date").apply(lambda x: x.to_dict("records")).to_dict()
    
    for dt in calendar:
        for r in exits_by_date.get(dt, []):
            key = (r["Ticker"], pd.Timestamp(r["Entry_Date"]))
            if key in open_pos:
                cash += r["Shares"] * r["Exit_Price"] * (1 - cfg.commission_bps/10000)
                del open_pos[key]
        for r in entries_by_date.get(dt, []):
            key = (r["Ticker"], pd.Timestamp(r["Entry_Date"]))
            open_pos[key] = r
            cash -= r["Shares"] * r["Entry_Price"] * (1 + cfg.commission_bps/10000)
            
        mkt_val = sum(get_close_on_or_before(data_cache[p["Ticker"]], dt, p["Entry_Price"]) * p["Shares"] for p in open_pos.values())
        equity = cash + mkt_val
        peak = max(peak, equity)
        dd = (equity/peak - 1)*100 if peak > 0 else 0
        rows.append({"Date": dt, "Cash": cash, "Equity": equity, "Drawdown_Pct": dd, "Positions": len(open_pos)})
    return pd.DataFrame(rows)

# ==========================================
# 9. Summaries (Full Version)
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
    sample_tickers = tickers[:200] 
    
    for t in tqdm(sample_tickers, desc="Loading Data"):
        data_cache[t] = get_data(t, "2014-01-01", "2026-03-01", cfg)
        
    cands = generate_candidates(sample_tickers, data_cache, spy, cfg)
    acc = accept_trades(cands, data_cache, cfg)
    eq = build_daily_equity_curve(acc, data_cache, spy, cfg)
    
    yearly_df = yearly_summary(acc, eq)
    monthly_df = monthly_summary(acc)
    overall = summarize_trades(acc, eq)
    
    return cands, acc, eq, yearly_df, monthly_df, overall

# ==========================================
# 11. Output Helpers (Full Version)
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
    print("ASCENDING TRIANGLE BACKTEST REPORT (v10.1)")
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
