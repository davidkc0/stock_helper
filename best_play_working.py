#!/usr/bin/env python3
"""
Best Options Play (Daily) - Starter
-----------------------------------
Chooses 1–3 *best* option ideas per day across:
  - CSP (Cash-Secured Put)
  - PCS (Put Credit Spread)
  - CONDOR (Iron Condor)

Decision logic blends:
  - Liquidity (bid/ask %, OI)
  - Volatility *rank* (HV rank proxy; IVR if your data vendor supports it)
  - Trend vs. range (RSI, ATR%, SMA bands)
  - Credit-to-risk & estimated POP
  - Your $42k account sizing rules (≤1% risk/spread, ≤40% collateral for CSP)

Data Provider: default yfinance (free, delayed). You can extend to Polygon/Tradier/IBKR.
Not investment advice. Use at your own risk.

Install deps:
    pip install yfinance pandas numpy scipy pyyaml

Run:
    python best_play.py --config best_play.yaml
"""
import argparse, math, sys, warnings
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from math import log, sqrt
from scipy.stats import norm

warnings.simplefilter("ignore", FutureWarning)

# -------------------------
# Config & Utilities
# -------------------------
@dataclass
class Config:
    tickers: List[str]
    account_size: float = 42000.0
    max_risk_per_trade_pct: float = 1.0   # % of account (spreads/condors max loss)
    max_collateral_pct: float = 40.0      # % of account (CSP total collateral cap)
    dte_min: int = 30
    dte_max: int = 60
    target_short_delta: float = 0.22
    delta_band: float = 0.08
    spread_width_low: float = 2.0
    spread_width_high: float = 5.0
    max_bid_ask_pct: float = 0.35
    min_open_interest: int = 50
    skip_earnings_within_days: int = 7
    risk_free_rate: float = 0.045
    take_profit_pct: float = 0.50
    stop_loss_mult: float = 2.0
    max_ideas: int = 3
    provider: str = "yfinance"
    polygon_api_key: Optional[str] = None
    tradier_access_token: Optional[str] = None

def mid(b,a):
    if pd.isna(b) or pd.isna(a) or (b<=0 and a<=0): return np.nan
    if b<=0: return a
    if a<=0: return b
    return (b+a)/2.0
def pct_spread(b,a):
    m = mid(b,a)
    return (a-b)/m if (m is not None and m>0) else np.inf
def bs_delta(S, K, r, iv, T, typ="C"):
    if S<=0 or K<=0 or iv<=0 or T<=0: return np.nan
    d1=(np.log(S/K)+(r+0.5*iv**2)*T)/(iv*np.sqrt(T))
    return norm.cdf(d1) if typ=="C" else norm.cdf(d1)-1.0
def years_to_exp(exp):
    T = (pd.to_datetime(exp, utc=True) - pd.Timestamp.utcnow()).days/365.0
    return max(T, 1/365)

def choose_width(spot, cfg: Config):
    return cfg.spread_width_low if spot<100 else cfg.spread_width_high

# ------------------ Provider (yfinance) ------------------
class YFProvider:
    def __init__(self):
        import yfinance as yf
        self.yf = yf
    def price(self, sym):
        t = self.yf.Ticker(sym)
        hist = t.history(period="1d")
        return float(hist["Close"].iloc[-1])
    def history(self, sym, period="1y"):
        t = self.yf.Ticker(sym)
        return t.history(period=period)
    def expirations(self, sym):
        return self.yf.Ticker(sym).options
    def chain(self, sym, exp):
        oc = self.yf.Ticker(sym).option_chain(exp)
        return oc.calls.copy(), oc.puts.copy()
    def earnings_date(self, sym):
        t = self.yf.Ticker(sym)
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                for col in cal.columns:
                    v = cal[col].dropna()
                    if not v.empty and isinstance(v.iloc[0], (np.datetime64, pd.Timestamp)):
                        return pd.to_datetime(v.iloc[0]).to_pydatetime()
        except Exception:
            pass
        return None

# ------------------ Indicators ------------------
def compute_indicators(hist: pd.DataFrame) -> Dict[str,float]:
    close = hist["Close"].dropna()
    if len(close)<60:
        return {"atrp":np.nan,"rsi":50,"sma20":close.iloc[-1] if len(close) else np.nan,
                "sma50":close.iloc[-1] if len(close) else np.nan, "hvr":50}
    high, low = hist["High"], hist["Low"]
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atrp = float(atr/close.iloc[-1])*100.0
    delta = close.diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    rs = (gain/(loss+1e-9)).iloc[-1]
    rsi = 100 - (100/(1+rs))
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    ret = np.log(close/close.shift())
    hv20 = ret.rolling(20).std()*np.sqrt(252)
    curr = hv20.iloc[-1]
    h = hv20.dropna()
    pct = 100.0 * (h<=curr).mean() if len(h)>20 else 50.0
    return {"atrp":float(atrp), "rsi":float(rsi), "sma20":float(sma20), "sma50":float(sma50), "hvr":float(pct)}

# ------------------ Strategy Generators ------------------
from dataclasses import dataclass
from typing import Optional

@dataclass
class Idea:
    symbol: str
    strategy: str
    expiration: str
    legs: list
    credit: float
    max_risk: float
    est_pop: float
    qty: int
    score: float
    notes: str

def liquidity_ok(row, cfg: Config):
    b,a = float(row.get("bid",np.nan)), float(row.get("ask",np.nan))
    if pd.isna(b) or pd.isna(a) or a<=0: return False
    if pct_spread(b,a) > cfg.max_bid_ask_pct: return False
    if int(row.get("openInterest",0)) < cfg.min_open_interest: return False
    return True

def nearest_by_delta(df, target_delta, band, S, r, T, opt_type):
    if df.empty: return None
    rows=[]
    for _,row in df.iterrows():
        iv=float(row.get("impliedVolatility", np.nan))
        if np.isnan(iv) or iv<=0: continue
        K=float(row["strike"])
        d = bs_delta(S,K,r,iv,T, "C" if opt_type=="C" else "P")
        rows.append((abs(d-target_delta), d, row))
    if not rows: return None
    rows.sort(key=lambda x: x[0])
    best = rows[0]
    return best if best[0] <= band else None

def gen_pcs(sym,S,calls,puts,exp,cfg: Config)->Optional[Idea]:
    T = years_to_exp(exp)
    res = nearest_by_delta(puts, target_delta=-cfg.target_short_delta, band=cfg.delta_band,
                           S=S, r=cfg.risk_free_rate, T=T, opt_type="P")
    if not res: return None
    _, short_delta, short_row = res
    if not liquidity_ok(short_row,cfg): return None
    shortK = float(short_row["strike"])
    width = choose_width(S,cfg)
    longK = max(0.5, shortK - width)
    long_row = puts.iloc[(puts["strike"]-longK).abs().argsort()[:1]].iloc[0]
    if not liquidity_ok(long_row,cfg): return None
    credit = mid(short_row["bid"],short_row["ask"]) - mid(long_row["bid"],long_row["ask"])
    if credit<=0 or pd.isna(credit): return None
    credit *= 100
    max_risk = (width*100 - credit)
    est_pop = max(0.0, 1.0 - abs(short_delta))
    max_risk_per_trade = cfg.account_size * (cfg.max_risk_per_trade_pct/100.0)
    qty = int(max(0, np.floor(max_risk_per_trade / max(1.0, max_risk))))
    if qty<1: qty = 1
    rr = credit / max(1.0, max_risk)
    score = rr*600 + est_pop*400
    legs=[
        {"side":"SELL","type":"PUT","strike":shortK,"exp":exp,"qty":qty},
        {"side":"BUY","type":"PUT","strike":float(long_row['strike']),"exp":exp,"qty":qty},
    ]
    notes=f"45DTE PCS ~{cfg.target_short_delta:.2f}Δ. TP +{cfg.take_profit_pct*100:.0f}%, stop {cfg.stop_loss_mult}x credit."
    return Idea(sym,"PCS",exp,legs,float(credit),float(max_risk),float(est_pop),qty,float(score),notes)

def gen_csp(sym,S,puts,exp,cfg: Config)->Optional[Idea]:
    T = years_to_exp(exp)
    res = nearest_by_delta(puts, target_delta=-cfg.target_short_delta, band=cfg.delta_band,
                           S=S, r=cfg.risk_free_rate, T=T, opt_type="P")
    if not res: return None
    _, dlt, row = res
    if not liquidity_ok(row,cfg): return None
    strike = float(row["strike"])
    credit = mid(row["bid"],row["ask"])
    if credit<=0 or pd.isna(credit): return None
    credit *= 100
    collateral = strike*100
    est_pop = max(0.0, 1.0 - abs(dlt))
    max_collateral = cfg.account_size * (cfg.max_collateral_pct/100.0)
    qty = int(max(0, np.floor(max_collateral / collateral)))
    if qty<1: return None
    dte = max(1,(pd.to_datetime(exp, utc=True)-pd.Timestamp.utcnow()).days)
    roc = (credit / collateral) * (365/dte)
    score = roc*500 + est_pop*300 + min(3,qty)*50
    legs=[{"side":"SELL","type":"PUT","strike":strike,"exp":exp,"qty":qty}]
    notes=f"Wheel entry. Collateral ~${collateral:.0f} x {qty}. TP +{cfg.take_profit_pct*100:.0f}% of premium; roll if tested."
    return Idea(sym,"CSP",exp,legs,float(credit),float(collateral),float(est_pop),qty,float(score),notes)

def gen_condor(sym,S,calls,puts,exp,cfg: Config)->Optional[Idea]:
    T = years_to_exp(exp)
    res_c = nearest_by_delta(calls, target_delta=cfg.target_short_delta, band=cfg.delta_band,
                             S=S, r=cfg.risk_free_rate, T=T, opt_type="C")
    res_p = nearest_by_delta(puts,  target_delta=-cfg.target_short_delta, band=cfg.delta_band,
                             S=S, r=cfg.risk_free_rate, T=T, opt_type="P")
    if not (res_c and res_p): return None
    _, dC, sc = res_c
    _, dP, sp = res_p
    if not (liquidity_ok(sc,cfg) and liquidity_ok(sp,cfg)): return None
    width = choose_width(S,cfg)
    lcK = float(sc["strike"]) + width
    lpK = float(sp["strike"]) - width
    lc = calls.iloc[(calls["strike"]-lcK).abs().argsort()[:1]].iloc[0]
    lp = puts.iloc[(puts["strike"]-lpK).abs().argsort()[:1]].iloc[0]
    if not (liquidity_ok(lc,cfg) and liquidity_ok(lp,cfg)): return None
    cr = (mid(sc["bid"],sc["ask"]) - mid(lc["bid"],lc["ask"])) + (mid(sp["bid"],sp["ask"]) - mid(lp["bid"],lp["ask"]))
    if cr<=0 or pd.isna(cr): return None
    credit = cr*100
    max_risk = width*100 - credit
    est_pop = max(0.0, 1.0 - (abs(dC)+abs(dP))/2.0)
    max_risk_per_trade = cfg.account_size * (cfg.max_risk_per_trade_pct/100.0)
    qty = int(max(0, np.floor(max_risk_per_trade / max(1.0, max_risk))))
    if qty<1: qty=1
    rr = credit / max(1.0, max_risk)
    score = rr*700 + est_pop*300
    legs=[
        {"side":"SELL","type":"CALL","strike":float(sc['strike']),"exp":exp,"qty":qty},
        {"side":"BUY","type":"CALL","strike":float(lc['strike']),"exp":exp,"qty":qty},
        {"side":"SELL","type":"PUT","strike":float(sp['strike']),"exp":exp,"qty":qty},
        {"side":"BUY","type":"PUT","strike":float(lp['strike']),"exp":exp,"qty":qty},
    ]
    notes=f"Range play. Width ${width:.0f}. TP +{cfg.take_profit_pct*100:.0f}%, stop {cfg.stop_loss_mult}x credit."
    return Idea(sym,"CONDOR",exp,legs,float(credit),float(max_risk),float(est_pop),qty,float(score),notes)

def pick_exp(exps, cfg: Config):
    center=(cfg.dte_min+cfg.dte_max)//2
    best=None; best_gap=9999
    now=pd.Timestamp.utcnow()
    for e in exps:
        try:
            dte=(pd.to_datetime(e, utc=True)-now).days
            if cfg.dte_min<=dte<=cfg.dte_max:
                gap=abs(dte-center)
                if gap<best_gap:
                    best=e; best_gap=gap
        except Exception: 
            continue
    return best

def strategy_bias_from_indicators(ind):
    hvr = ind.get("hvr",50); rsi = ind.get("rsi",50); atrp= ind.get("atrp",2)
    w = {"CSP":1.0,"PCS":1.0,"CONDOR":1.0}
    if hvr>=60:
        w["CONDOR"]*=1.2; w["PCS"]*=1.15; w["CSP"]*=1.1
    elif hvr<=25:
        w["CONDOR"]*=0.6; w["PCS"]*=0.8; w["CSP"]*=0.9
    if 45<=rsi<=55 and atrp<2.5:
        w["CONDOR"]*=1.25
    if rsi>55:
        w["CSP"]*=1.15; w["PCS"]*=1.05
    if rsi<45:
        w["CONDOR"]*=1.05
    return w

def main():
    import argparse, json, yaml, yfinance as yf
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="best_play.yaml")
    ap.add_argument("--out", default="best_play_output.json")
    args=ap.parse_args()
    with open(args.config) as f: cfg = Config(**yaml.safe_load(f))

    provider = YFProvider()
    all_ideas=[]
    for sym in cfg.tickers:
        try:
            price = provider.price(sym)
            hist = provider.history(sym, period="1y")
            ind  = compute_indicators(hist)
            exps = provider.expirations(sym)
            exp  = pick_exp(exps, cfg)
            if exp is None:
                continue
            calls, puts = provider.chain(sym, exp)
            calls_filtered = calls[(calls["openInterest"]>=cfg.min_open_interest)]
            puts_filtered  = puts[(puts["openInterest"]>=cfg.min_open_interest)]
            if calls_filtered.empty or puts_filtered.empty: 
                continue
            pcs = gen_pcs(sym, price, calls_filtered, puts_filtered, exp, cfg)
            csp = gen_csp(sym, price, puts_filtered, exp, cfg)
            ic  = gen_condor(sym, price, calls_filtered, puts_filtered, exp, cfg)
            ideas=[x for x in [pcs,csp,ic] if x is not None]
            if not ideas: 
                continue
            bias = strategy_bias_from_indicators(ind)
            for idea in ideas:
                idea.score *= bias.get(idea.strategy,1.0)
                if idea.strategy!="CSP":
                    max_per_trade = cfg.account_size*(cfg.max_risk_per_trade_pct/100.0)
                    if idea.max_risk*idea.qty > max_per_trade*1.2:
                        idea.score *= 0.8
            ideas.sort(key=lambda x: x.score, reverse=True)
            best_for_symbol = ideas[0]
            tele = {"price":float(price), **ind}
            best_for_symbol_dict = asdict(best_for_symbol)
            best_for_symbol_dict["_telemetry"]=tele
            all_ideas.append(best_for_symbol_dict)
        except Exception as e:
            print(f"[{sym}] error: {e}")
            continue
    all_ideas.sort(key=lambda d: d["score"], reverse=True)
    top = all_ideas[:cfg.max_ideas]
    with open(args.out,"w") as f:
        json.dump(top, f, indent=2)
    if not top:
        print("No tradable ideas today with current filters.")
        return
    print("="*80)
    print("Best Options Plays (Top {}):".format(cfg.max_ideas))
    print("="*80)
    for i,d in enumerate(top,1):
        print(f"[{i}] {d['symbol']}  | {d['strategy']}  | {d['expiration']}  | Score {d['score']:.1f}")
        print(f"  Price: ${d['_telemetry']['price']:.2f}  HVR:{d['_telemetry']['hvr']:.0f}  RSI:{d['_telemetry']['rsi']:.0f}  ATR%:{d['_telemetry']['atrp']:.2f}")
        print(f"  Credit/contract: ${d['credit']:.2f}   Max risk/contract: ${d['max_risk']:.2f}   Est POP: {d['est_pop']*100:.1f}%")
        print("  Legs:")
        for leg in d["legs"]:
            print("    {side:>4} {type} {strike:.2f} x{qty}  @ {exp}".format(**leg))
        print("  Notes:", d["notes"])
        print("-"*80)

if __name__=="__main__":
    main()
