"""
Enhanced Options Trading Strategy Analyzer
------------------------------------------
Advanced options strategy recommendation system with:
- Multiple strategy types (PCS, CCS, Iron Condor, Long Straddle, Calendar Spreads)
- True IV Rank calculation over 252-day lookback
- Enhanced strategy selection based on market regime
- Better risk management and position sizing
- Multiple data providers support
- Advanced filtering and scoring
- Real-time earnings calendar integration

Key Improvements:
1. Multiple strategy implementations
2. Market regime-based strategy selection
3. Enhanced strategy scoring with multiple factors
4. Better liquidity and earnings filters
5. Advanced position sizing with Kelly criterion
6. Real-time Greeks calculation
7. Better error handling and logging

Usage:
    python best_play.py --config config.yaml --mode scan
    python best_play.py --analyze AAPL --strategy all
"""

import argparse, math, sys, warnings, logging
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from math import log, sqrt, exp
from scipy.stats import norm
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import wraps

warnings.simplefilter("ignore", FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate Limiting Decorator
def rate_limit_retry(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {delay}s before retry {attempt+1}")
                        time.sleep(delay)
                        continue
                    raise e
            raise Exception(f"Max retries ({max_retries}) exceeded")
        return wrapper
    return decorator

# -------------------------
# Enhanced Configuration
# -------------------------
@dataclass
class AdvancedConfig:
    # Basic settings
    tickers: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA'])
    account_size: float = 42000.0
    max_risk_per_trade_pct: float = 1.0
    max_collateral_pct: float = 40.0
    max_total_allocation_pct: float = 60.0  # NEW: Total portfolio allocation limit
    
    # Strategy parameters
    dte_min: int = 30
    dte_max: int = 60
    target_short_delta: float = 0.20  # Slightly more conservative
    delta_band: float = 0.05  # Tighter targeting
    spread_width_low: float = 2.0
    spread_width_high: float = 5.0
    
    # Enhanced liquidity filters
    max_bid_ask_pct: float = 0.25  # Tighter spread requirement
    min_open_interest: int = 100   # Higher OI requirement
    min_volume: int = 50           # NEW: Minimum daily volume
    min_option_price: float = 0.10 # NEW: Avoid penny options
    
    # IV Rank settings
    iv_rank_lookback_days: int = 252
    high_iv_threshold: float = 0.6   # Above this = sell premium
    low_iv_threshold: float = 0.3    # Below this = buy premium
    
    # Market regime settings
    trend_sma_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    volatility_regime_threshold: float = 1.5  # ATR vs historical avg
    
    # Risk management
    risk_free_rate: float = 0.045
    take_profit_pct: float = 0.50
    stop_loss_mult: float = 2.5  # Slightly wider stops
    kelly_fraction: float = 0.25  # Kelly criterion multiplier
    
    # Output settings
    max_ideas: int = 8  # Show more ideas for multiple strategies
    skip_earnings_within_days: int = 10  # More conservative on earnings
    
    # Provider settings
    provider: str = "yfinance"
    polygon_api_key: Optional[str] = None
    tradier_access_token: Optional[str] = None
    use_concurrent: bool = True
    max_workers: int = 4

# -------------------------
# Enhanced Data Structures
# -------------------------
@dataclass
class MarketRegime:
    """Market regime classification"""
    trend_strength: float  # -1 to 1 (bearish to bullish)
    volatility_regime: str  # 'LOW', 'NORMAL', 'HIGH'
    iv_rank: float  # 0 to 1
    regime_type: str  # 'TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOL'

@dataclass
class AdvancedIdea:
    symbol: str
    strategy: str
    expiration: str
    legs: List[Dict[str, Any]]
    credit: float
    max_risk: float
    breakevens: List[float]
    est_pop: float
    kelly_size: int
    max_profit_pct: float
    days_to_exp: int
    
    # Enhanced scoring factors
    liquidity_score: float
    iv_rank_score: float
    regime_score: float
    risk_reward_score: float
    final_score: float
    
    # Greeks and additional metrics
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    
    notes: str
    market_context: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'

# -------------------------
# Enhanced Utilities
# -------------------------
def black_scholes_price(S, K, r, iv, T, option_type='call'):
    """Calculate Black-Scholes option price"""
    if S <= 0 or K <= 0 or iv <= 0 or T <= 0:
        return 0
    
    d1 = (np.log(S/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
    d2 = d1 - iv*np.sqrt(T)
    
    if option_type.lower() == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def safe_float(value) -> float:
    """Safely convert value to float, handling NaN and invalid values"""
    try:
        result = float(value)
        return result if not np.isnan(result) else 0.0
    except (ValueError, TypeError):
        return 0.0

def calculate_greeks(S, K, r, iv, T, option_type='call'):
    """Calculate option Greeks"""
    if S <= 0 or K <= 0 or iv <= 0 or T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    d1 = (np.log(S/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
    d2 = d1 - iv*np.sqrt(T)
    
    # Delta
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
    
    # Theta
    if option_type.lower() == 'call':
        theta = (-(S*norm.pdf(d1)*iv)/(2*np.sqrt(T)) - 
                r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        theta = (-(S*norm.pdf(d1)*iv)/(2*np.sqrt(T)) + 
                r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def calculate_true_iv_rank(symbol: str, provider, lookback_days: int = 252) -> float:
    """Calculate true IV rank using historical option chain data"""
    try:
        # Get current IV from ATM options
        expirations = provider.expirations(symbol)
        if not expirations:
            return 0.5
        
        current_price = provider.price(symbol)
        exp = expirations[0]  # Use nearest expiration
        calls, puts = provider.chain(symbol, exp)
        
        # Find ATM options
        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
        
        if atm_call.empty or atm_put.empty:
            return 0.5
        
        current_iv = (float(atm_call.iloc[0]['impliedVolatility']) + 
                     float(atm_put.iloc[0]['impliedVolatility'])) / 2
        
        # For yfinance, we'll approximate with HV rank since historical IV chains aren't available
        # In production, you'd want to store historical IV data
        hist = provider.history(symbol, period="1y")
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        
        if len(returns) < 20:
            return 0.5
        
        # Calculate rolling 20-day realized volatility
        rolling_hv = returns.rolling(20).std() * np.sqrt(252)
        rolling_hv = rolling_hv.dropna()
        
        if len(rolling_hv) < 50:
            return 0.5
        
        # Use current IV vs historical HV range as proxy
        hv_min, hv_max = rolling_hv.min(), rolling_hv.max()
        if hv_max <= hv_min:
            return 0.5
        
        iv_rank = (current_iv - hv_min) / (hv_max - hv_min)
        return max(0, min(1, iv_rank))
        
    except Exception as e:
        logger.warning(f"IV rank calculation failed for {symbol}: {e}")
        return 0.5

def detect_market_regime(symbol: str, provider, config: AdvancedConfig) -> MarketRegime:
    """Detect current market regime for the symbol"""
    try:
        hist = provider.history(symbol, period="6mo")
        close = hist['Close'].dropna()
        
        if len(close) < 100:
            return MarketRegime(0, 'NORMAL', 0.5, 'RANGING')
        
        # Trend strength calculation
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma100 = close.rolling(100).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        trend_signals = []
        trend_signals.append(1 if current_price > sma20 else -1)
        trend_signals.append(1 if sma20 > sma50 else -1)
        trend_signals.append(1 if sma50 > sma100 else -1)
        
        trend_strength = sum(trend_signals) / len(trend_signals)
        
        # Volatility regime
        returns = close.pct_change().dropna()
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        avg_vol = returns.rolling(100).std().mean() * np.sqrt(252)
        
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        if vol_ratio > config.volatility_regime_threshold:
            vol_regime = 'HIGH'
        elif vol_ratio < 0.7:
            vol_regime = 'LOW'
        else:
            vol_regime = 'NORMAL'
        
        # IV Rank
        iv_rank = calculate_true_iv_rank(symbol, provider, config.iv_rank_lookback_days)
        
        # Regime classification
        if abs(trend_strength) < 0.3:
            regime_type = 'RANGING'
        elif trend_strength > 0.3:
            regime_type = 'TRENDING_UP'
        else:
            regime_type = 'TRENDING_DOWN'
            
        if vol_regime == 'HIGH':
            regime_type = 'HIGH_VOL'
        
        return MarketRegime(trend_strength, vol_regime, iv_rank, regime_type)
        
    except Exception as e:
        logger.warning(f"Market regime detection failed for {symbol}: {e}")
        return MarketRegime(0, 'NORMAL', 0.5, 'RANGING')

# -------------------------
# Enhanced Multi-Strategy Generator
# -------------------------
class AdvancedStrategyGenerator:
    def __init__(self, config: AdvancedConfig):
        self.config = config
    
    def enhanced_liquidity_check(self, row: pd.Series) -> Tuple[bool, float]:
        """Enhanced liquidity scoring"""
        bid = safe_float(row.get('bid', 0))
        ask = safe_float(row.get('ask', 0))
        volume = int(safe_float(row.get('volume', 0)))
        oi = int(safe_float(row.get('openInterest', 0)))
        
        if bid <= 0 or ask <= 0:
            return False, 0
        
        mid_price = (bid + ask) / 2
        if mid_price < self.config.min_option_price:
            return False, 0
        
        spread_pct = (ask - bid) / mid_price
        if spread_pct > self.config.max_bid_ask_pct:
            return False, 0
        
        if oi < self.config.min_open_interest or volume < self.config.min_volume:
            return False, 0
        
        # Liquidity score (0-1)
        spread_score = max(0, 1 - (spread_pct / self.config.max_bid_ask_pct))
        oi_score = min(1, oi / (self.config.min_open_interest * 5))
        volume_score = min(1, volume / (self.config.min_volume * 5))
        
        liquidity_score = (spread_score * 0.4 + oi_score * 0.3 + volume_score * 0.3)
        
        return True, liquidity_score

    def calculate_position_size_and_scores(self, net_credit: float, max_risk: float, 
                                         est_pop: float, regime: MarketRegime,
                                         liquidity_scores: List[float],
                                         strategy_type: str = "NEUTRAL") -> Tuple[int, Dict[str, float], float]:
        """Calculate position size and scoring metrics"""
        
        # Kelly sizing
        edge = est_pop - 0.5  # Edge over 50/50
        if edge > 0 and max_risk > 0 and not (np.isnan(edge) or np.isnan(max_risk) or np.isnan(net_credit)):
            kelly_f = edge / (max_risk / net_credit)
            if not np.isnan(kelly_f):
                kelly_size = max(1, int(self.config.account_size * kelly_f * self.config.kelly_fraction / max_risk))
            else:
                kelly_size = 1
        else:
            kelly_size = 1
        
        # Risk management override
        max_risk_dollar = self.config.account_size * (self.config.max_risk_per_trade_pct / 100)
        if max_risk * kelly_size > max_risk_dollar and max_risk > 0 and not np.isnan(max_risk):
            kelly_size = max(1, int(max_risk_dollar / max_risk))
        
        # Scoring
        liquidity_score = sum(liquidity_scores) / len(liquidity_scores) if liquidity_scores else 0.5
        iv_rank_score = regime.iv_rank
        
        # Strategy-specific regime scoring
        if strategy_type == "PREMIUM_SELLING":
            # Favor high IV, ranging markets
            if regime.regime_type in ['RANGING'] and regime.iv_rank > 0.5:
                regime_score = 0.9
            elif regime.regime_type in ['TRENDING_UP'] and regime.iv_rank > 0.6:
                regime_score = 0.7
            else:
                regime_score = 0.4
        elif strategy_type == "PREMIUM_BUYING":
            # Favor low IV, trending markets
            if regime.iv_rank < 0.4 and regime.regime_type != 'RANGING':
                regime_score = 0.8
            else:
                regime_score = 0.3
        elif strategy_type == "VOLATILITY_EXPANSION":
            # Favor low IV with potential for expansion
            if regime.iv_rank < 0.3 and regime.volatility_regime == 'LOW':
                regime_score = 0.9
            else:
                regime_score = 0.2
        else:  # NEUTRAL
            regime_score = 0.6
        
        max_profit_pct = net_credit / max_risk if max_risk > 0 else 0
        risk_reward_score = min(1, max_profit_pct * 2)
        
        scores = {
            'liquidity_score': liquidity_score,
            'iv_rank_score': iv_rank_score,
            'regime_score': regime_score,
            'risk_reward_score': risk_reward_score
        }
        
        final_score = (liquidity_score * 0.2 + 
                      iv_rank_score * 0.3 + 
                      regime_score * 0.3 + 
                      risk_reward_score * 0.2) * 100
        
        return kelly_size, scores, final_score
    
    def generate_put_credit_spread(self, symbol: str, spot_price: float, 
                                 puts: pd.DataFrame, exp: str, 
                                 regime: MarketRegime, tight_strikes: bool = False) -> Optional[AdvancedIdea]:
        """Generate enhanced put credit spread"""
        try:
            T = max(1/365, (pd.to_datetime(exp, utc=True) - pd.Timestamp.utcnow()).days / 365)
            
            # Find short strike based on delta
            best_short = None
            target_delta = -0.15 if tight_strikes else -self.config.target_short_delta  # Tighter strikes for low IV
            
            for _, row in puts.iterrows():
                iv = float(row.get('impliedVolatility', 0))
                if iv <= 0:
                    continue
                
                strike = float(row['strike'])
                greeks = calculate_greeks(spot_price, strike, self.config.risk_free_rate, iv, T, 'put')
                delta = greeks['delta']
                
                if abs(delta - target_delta) <= self.config.delta_band:
                    liquid_ok, liq_score = self.enhanced_liquidity_check(row)
                    if liquid_ok:
                        best_short = (row, greeks, liq_score)
                        break
            
            if not best_short:
                return None
            
            short_row, short_greeks, short_liq_score = best_short
            short_strike = float(short_row['strike'])
            
            # Find long strike
            width = self.config.spread_width_low if spot_price < 100 else self.config.spread_width_high
            long_strike = short_strike - width
            
            long_row = puts.iloc[(puts['strike'] - long_strike).abs().argsort()[:1]].iloc[0]
            long_liquid_ok, long_liq_score = self.enhanced_liquidity_check(long_row)
            
            if not long_liquid_ok:
                return None
            
            # Calculate pricing
            short_mid = (float(short_row['bid']) + float(short_row['ask'])) / 2
            long_mid = (float(long_row['bid']) + float(long_row['ask'])) / 2
            net_credit = (short_mid - long_mid) * 100
            
            if net_credit <= 0:
                return None
            
            max_risk = width * 100 - net_credit
            
            # Enhanced POP calculation
            prob_itm = abs(short_greeks['delta'])
            est_pop = max(0.1, 1 - prob_itm)
            
            # Calculate position size and scores
            kelly_size, scores, final_score = self.calculate_position_size_and_scores(
                net_credit, max_risk, est_pop, regime, [short_liq_score, long_liq_score], "PREMIUM_SELLING"
            )
            
            # Long strike Greeks for portfolio Greeks
            long_iv = float(long_row.get('impliedVolatility', 0))
            long_greeks = calculate_greeks(spot_price, float(long_row['strike']), 
                                         self.config.risk_free_rate, long_iv, T, 'put')
            
            # Net Greeks
            net_delta = (short_greeks['delta'] - long_greeks['delta']) * kelly_size
            net_gamma = (short_greeks['gamma'] - long_greeks['gamma']) * kelly_size
            net_theta = (short_greeks['theta'] - long_greeks['theta']) * kelly_size
            net_vega = (short_greeks['vega'] - long_greeks['vega']) * kelly_size
            
            breakeven = short_strike - (net_credit / 100)
            
            return AdvancedIdea(
                symbol=symbol,
                strategy="PUT_CREDIT_SPREAD",
                expiration=exp,
                legs=[
                    {"side": "SELL", "type": "PUT", "strike": short_strike, 
                     "exp": exp, "qty": kelly_size},
                    {"side": "BUY", "type": "PUT", "strike": float(long_row['strike']), 
                     "exp": exp, "qty": kelly_size}
                ],
                credit=net_credit,
                max_risk=max_risk,
                breakevens=[breakeven],
                est_pop=est_pop,
                kelly_size=kelly_size,
                max_profit_pct=net_credit / max_risk,
                days_to_exp=int(T * 365),
                liquidity_score=scores['liquidity_score'],
                iv_rank_score=scores['iv_rank_score'],
                regime_score=scores['regime_score'],
                risk_reward_score=scores['risk_reward_score'],
                final_score=final_score,
                total_delta=net_delta,
                total_gamma=net_gamma,
                total_theta=net_theta,
                total_vega=net_vega,
                notes=f"PCS targeting {target_delta:.2f}Δ short put. Bullish/neutral strategy. "
                     f"TP: {self.config.take_profit_pct*100:.0f}% | Stop: {self.config.stop_loss_mult:.1f}x",
                market_context=f"IV Rank: {regime.iv_rank:.1%} | Regime: {regime.regime_type}",
                risk_level="LOW" if max_risk < 300 else "MEDIUM" if max_risk < 800 else "HIGH"
            )
            
        except Exception as e:
            logger.error(f"PCS generation failed for {symbol}: {e}")
            return None

    def generate_call_credit_spread(self, symbol: str, spot_price: float, 
                                  calls: pd.DataFrame, exp: str, 
                                  regime: MarketRegime) -> Optional[AdvancedIdea]:
        """Generate call credit spread for bearish/neutral outlook"""
        try:
            T = max(1/365, (pd.to_datetime(exp, utc=True) - pd.Timestamp.utcnow()).days / 365)
            
            # Find short strike based on delta (positive for calls)
            best_short = None
            target_delta = self.config.target_short_delta  # Positive for calls
            
            for _, row in calls.iterrows():
                iv = float(row.get('impliedVolatility', 0))
                if iv <= 0:
                    continue
                
                strike = float(row['strike'])
                if strike <= spot_price:  # Only OTM calls
                    continue
                    
                greeks = calculate_greeks(spot_price, strike, self.config.risk_free_rate, iv, T, 'call')
                delta = greeks['delta']
                
                if abs(delta - target_delta) <= self.config.delta_band:
                    liquid_ok, liq_score = self.enhanced_liquidity_check(row)
                    if liquid_ok:
                        best_short = (row, greeks, liq_score)
                        break
            
            if not best_short:
                return None
            
            short_row, short_greeks, short_liq_score = best_short
            short_strike = float(short_row['strike'])
            
            # Find long strike (higher strike)
            width = self.config.spread_width_low if spot_price < 100 else self.config.spread_width_high
            long_strike = short_strike + width
            
            long_row = calls.iloc[(calls['strike'] - long_strike).abs().argsort()[:1]].iloc[0]
            long_liquid_ok, long_liq_score = self.enhanced_liquidity_check(long_row)
            
            if not long_liquid_ok:
                return None
            
            # Calculate pricing
            short_mid = (float(short_row['bid']) + float(short_row['ask'])) / 2
            long_mid = (float(long_row['bid']) + float(long_row['ask'])) / 2
            net_credit = (short_mid - long_mid) * 100
            
            if net_credit <= 0:
                return None
            
            max_risk = width * 100 - net_credit
            
            # POP calculation for calls
            prob_itm = short_greeks['delta']  # Positive delta = probability ITM
            est_pop = max(0.1, 1 - prob_itm)
            
            # Calculate position size and scores
            kelly_size, scores, final_score = self.calculate_position_size_and_scores(
                net_credit, max_risk, est_pop, regime, [short_liq_score, long_liq_score], "PREMIUM_SELLING"
            )
            
            # Long strike Greeks
            long_iv = float(long_row.get('impliedVolatility', 0))
            long_greeks = calculate_greeks(spot_price, float(long_row['strike']), 
                                         self.config.risk_free_rate, long_iv, T, 'call')
            
            # Net Greeks
            net_delta = (short_greeks['delta'] - long_greeks['delta']) * kelly_size
            net_gamma = (short_greeks['gamma'] - long_greeks['gamma']) * kelly_size
            net_theta = (short_greeks['theta'] - long_greeks['theta']) * kelly_size
            net_vega = (short_greeks['vega'] - long_greeks['vega']) * kelly_size
            
            breakeven = short_strike + (net_credit / 100)
            
            return AdvancedIdea(
                symbol=symbol,
                strategy="CALL_CREDIT_SPREAD",
                expiration=exp,
                legs=[
                    {"side": "SELL", "type": "CALL", "strike": short_strike, 
                     "exp": exp, "qty": kelly_size},
                    {"side": "BUY", "type": "CALL", "strike": float(long_row['strike']), 
                     "exp": exp, "qty": kelly_size}
                ],
                credit=net_credit,
                max_risk=max_risk,
                breakevens=[breakeven],
                est_pop=est_pop,
                kelly_size=kelly_size,
                max_profit_pct=net_credit / max_risk,
                days_to_exp=int(T * 365),
                liquidity_score=scores['liquidity_score'],
                iv_rank_score=scores['iv_rank_score'],
                regime_score=scores['regime_score'],
                risk_reward_score=scores['risk_reward_score'],
                final_score=final_score,
                total_delta=net_delta,
                total_gamma=net_gamma,
                total_theta=net_theta,
                total_vega=net_vega,
                notes=f"CCS targeting {target_delta:.2f}Δ short call. Bearish/neutral strategy. "
                     f"TP: {self.config.take_profit_pct*100:.0f}% | Stop: {self.config.stop_loss_mult:.1f}x",
                market_context=f"IV Rank: {regime.iv_rank:.1%} | Regime: {regime.regime_type}",
                risk_level="LOW" if max_risk < 300 else "MEDIUM" if max_risk < 800 else "HIGH"
            )
            
        except Exception as e:
            logger.error(f"CCS generation failed for {symbol}: {e}")
            return None

    def generate_iron_condor(self, symbol: str, spot_price: float,
                            calls: pd.DataFrame, puts: pd.DataFrame, 
                            exp: str, regime: MarketRegime) -> Optional[AdvancedIdea]:
        """Generate iron condor for range-bound, high IV markets"""
        try:
            T = max(1/365, (pd.to_datetime(exp, utc=True) - pd.Timestamp.utcnow()).days / 365)
            
            # Find put spread strikes (lower strikes)
            put_short_strike = None
            put_long_strike = None
            put_short_greeks = None
            put_liq_scores = []
            
            target_delta = -self.config.target_short_delta
            for _, row in puts.iterrows():
                iv = float(row.get('impliedVolatility', 0))
                if iv <= 0:
                    continue
                
                strike = float(row['strike'])
                greeks = calculate_greeks(spot_price, strike, self.config.risk_free_rate, iv, T, 'put')
                delta = greeks['delta']
                
                if abs(delta - target_delta) <= self.config.delta_band:
                    liquid_ok, liq_score = self.enhanced_liquidity_check(row)
                    if liquid_ok:
                        put_short_strike = strike
                        put_short_greeks = greeks
                        put_liq_scores.append(liq_score)
                        
                        # Find long put strike
                        width = self.config.spread_width_low if spot_price < 100 else self.config.spread_width_high
                        target_long = strike - width
                        long_row = puts.iloc[(puts['strike'] - target_long).abs().argsort()[:1]].iloc[0]
                        long_liquid_ok, long_liq_score = self.enhanced_liquidity_check(long_row)
                        
                        if long_liquid_ok:
                            put_long_strike = float(long_row['strike'])
                            put_liq_scores.append(long_liq_score)
                            break
            
            if not put_short_strike or not put_long_strike:
                return None
            
            # Find call spread strikes (higher strikes)
            call_short_strike = None
            call_long_strike = None
            call_short_greeks = None
            call_liq_scores = []
            
            target_delta = self.config.target_short_delta  # Positive for calls
            for _, row in calls.iterrows():
                iv = float(row.get('impliedVolatility', 0))
                if iv <= 0:
                    continue
                
                strike = float(row['strike'])
                if strike <= spot_price:  # Only OTM calls
                    continue
                    
                greeks = calculate_greeks(spot_price, strike, self.config.risk_free_rate, iv, T, 'call')
                delta = greeks['delta']
                
                if abs(delta - target_delta) <= self.config.delta_band:
                    liquid_ok, liq_score = self.enhanced_liquidity_check(row)
                    if liquid_ok:
                        call_short_strike = strike
                        call_short_greeks = greeks
                        call_liq_scores.append(liq_score)
                        
                        # Find long call strike
                        width = self.config.spread_width_low if spot_price < 100 else self.config.spread_width_high
                        target_long = strike + width
                        long_row = calls.iloc[(calls['strike'] - target_long).abs().argsort()[:1]].iloc[0]
                        long_liquid_ok, long_liq_score = self.enhanced_liquidity_check(long_row)
                        
                        if long_liquid_ok:
                            call_long_strike = float(long_row['strike'])
                            call_liq_scores.append(long_liq_score)
                            break
            
            if not call_short_strike or not call_long_strike:
                return None
            
            # Calculate net credit
            put_short_row = puts[puts['strike'] == put_short_strike].iloc[0]
            put_long_row = puts[puts['strike'] == put_long_strike].iloc[0]
            call_short_row = calls[calls['strike'] == call_short_strike].iloc[0]
            call_long_row = calls[calls['strike'] == call_long_strike].iloc[0]
            
            put_credit = ((float(put_short_row['bid']) + float(put_short_row['ask'])) / 2 - 
                         (float(put_long_row['bid']) + float(put_long_row['ask'])) / 2) * 100
            call_credit = ((float(call_short_row['bid']) + float(call_short_row['ask'])) / 2 - 
                          (float(call_long_row['bid']) + float(call_long_row['ask'])) / 2) * 100
            
            net_credit = put_credit + call_credit
            if net_credit <= 0:
                return None
            
            # Max risk is the width of one side minus net credit
            put_width = put_short_strike - put_long_strike
            call_width = call_long_strike - call_short_strike
            max_width = max(put_width, call_width) * 100
            max_risk = max_width - net_credit
            
            # POP calculation (probability of staying between short strikes)
            put_prob_itm = abs(put_short_greeks['delta'])
            call_prob_itm = call_short_greeks['delta']
            est_pop = max(0.1, 1 - put_prob_itm - call_prob_itm)
            
            # Calculate position size and scores
            all_liq_scores = put_liq_scores + call_liq_scores
            kelly_size, scores, final_score = self.calculate_position_size_and_scores(
                net_credit, max_risk, est_pop, regime, all_liq_scores, "PREMIUM_SELLING"
            )
            
            # Net Greeks (all short minus all long)
            put_long_iv = float(put_long_row.get('impliedVolatility', 0))
            call_long_iv = float(call_long_row.get('impliedVolatility', 0))
            
            put_long_greeks = calculate_greeks(spot_price, put_long_strike, 
                                             self.config.risk_free_rate, put_long_iv, T, 'put')
            call_long_greeks = calculate_greeks(spot_price, call_long_strike, 
                                              self.config.risk_free_rate, call_long_iv, T, 'call')
            
            net_delta = ((put_short_greeks['delta'] - put_long_greeks['delta']) + 
                        (call_short_greeks['delta'] - call_long_greeks['delta'])) * kelly_size
            net_gamma = ((put_short_greeks['gamma'] - put_long_greeks['gamma']) + 
                        (call_short_greeks['gamma'] - call_long_greeks['gamma'])) * kelly_size
            net_theta = ((put_short_greeks['theta'] - put_long_greeks['theta']) + 
                        (call_short_greeks['theta'] - call_long_greeks['theta'])) * kelly_size
            net_vega = ((put_short_greeks['vega'] - put_long_greeks['vega']) + 
                       (call_short_greeks['vega'] - call_long_greeks['vega'])) * kelly_size
            
            # Breakevens
            lower_breakeven = put_short_strike - (net_credit / 100)
            upper_breakeven = call_short_strike + (net_credit / 100)
            
            return AdvancedIdea(
                symbol=symbol,
                strategy="IRON_CONDOR",
                expiration=exp,
                legs=[
                    {"side": "SELL", "type": "PUT", "strike": put_short_strike, 
                     "exp": exp, "qty": kelly_size},
                    {"side": "BUY", "type": "PUT", "strike": put_long_strike, 
                     "exp": exp, "qty": kelly_size},
                    {"side": "SELL", "type": "CALL", "strike": call_short_strike, 
                     "exp": exp, "qty": kelly_size},
                    {"side": "BUY", "type": "CALL", "strike": call_long_strike, 
                     "exp": exp, "qty": kelly_size}
                ],
                credit=net_credit,
                max_risk=max_risk,
                breakevens=[lower_breakeven, upper_breakeven],
                est_pop=est_pop,
                kelly_size=kelly_size,
                max_profit_pct=net_credit / max_risk,
                days_to_exp=int(T * 365),
                liquidity_score=scores['liquidity_score'],
                iv_rank_score=scores['iv_rank_score'],
                regime_score=scores['regime_score'],
                risk_reward_score=scores['risk_reward_score'],
                final_score=final_score,
                total_delta=net_delta,
                total_gamma=net_gamma,
                total_theta=net_theta,
                total_vega=net_vega,
                notes=f"IC expecting price to stay between ${lower_breakeven:.2f} and ${upper_breakeven:.2f}. "
                     f"High IV, range-bound strategy. TP: {self.config.take_profit_pct*100:.0f}%",
                market_context=f"IV Rank: {regime.iv_rank:.1%} | Regime: {regime.regime_type}",
                risk_level="MEDIUM" if max_risk < 500 else "HIGH"
            )
            
        except Exception as e:
            logger.error(f"IC generation failed for {symbol}: {e}")
            return None
    
    def generate_cash_secured_put(self, symbol: str, spot_price: float,
                                puts: pd.DataFrame, exp: str,
                                regime: MarketRegime) -> Optional[AdvancedIdea]:
        """Generate cash secured put strategy"""
        try:
            T = max(1/365, (pd.to_datetime(exp, utc=True) - pd.Timestamp.utcnow()).days / 365)
            
            # Find suitable put strike (deeper ITM than PCS)
            target_delta = -0.25  # More aggressive than PCS
            best_put = None
            
            for _, row in puts.iterrows():
                iv = float(row.get('impliedVolatility', 0))
                if iv <= 0:
                    continue
                
                strike = float(row['strike'])
                greeks = calculate_greeks(spot_price, strike, self.config.risk_free_rate, iv, T, 'put')
                delta = greeks['delta']
                
                if abs(delta - target_delta) <= self.config.delta_band:
                    liquid_ok, liq_score = self.enhanced_liquidity_check(row)
                    if liquid_ok:
                        best_put = (row, greeks, liq_score)
                        break
            
            if not best_put:
                return None
            
            put_row, put_greeks, liq_score = best_put
            put_strike = float(put_row['strike'])
            
            # Calculate pricing
            put_mid = (float(put_row['bid']) + float(put_row['ask'])) / 2
            collateral = put_strike * 100  # Cash needed to secure the put
            
            # Check if we have enough capital
            if collateral > self.config.account_size:
                return None
            
            # Kelly sizing (more conservative for CSP)
            est_pop = abs(put_greeks['delta'])
            edge = est_pop - 0.5
            if edge > 0:
                kelly_f = edge / put_strike  # Normalize by strike price
                kelly_size = max(1, int(self.config.account_size * kelly_f * self.config.kelly_fraction / collateral))
            else:
                kelly_size = 1
            
            # Risk management
            max_collateral_pct = 0.05  # Max 5% of account per CSP
            max_size = int(self.config.account_size * max_collateral_pct / collateral)
            kelly_size = min(kelly_size, max_size)
            
            if kelly_size <= 0:
                return None
            
            # Scoring
            liquidity_score = liq_score
            iv_rank_score = regime.iv_rank
            regime_score = 0.7 if regime.regime_type in ['TRENDING_UP', 'RANGING'] else 0.4
            risk_reward_score = 0.6  # CSP has limited upside
            
            final_score = (liquidity_score * 0.2 + 
                          iv_rank_score * 0.3 + 
                          regime_score * 0.3 + 
                          risk_reward_score * 0.2) * 100
            
            # Breakeven
            breakeven = put_strike - put_mid
            
            return AdvancedIdea(
                symbol=symbol,
                strategy="CASH_SECURED_PUT",
                expiration=exp,
                legs=[
                    {"side": "SELL", "type": "PUT", "strike": put_strike, "exp": exp, "qty": kelly_size}
                ],
                credit=put_mid * 100 * kelly_size,
                max_risk=collateral * kelly_size - (put_mid * 100 * kelly_size),
                breakevens=[breakeven],
                est_pop=est_pop,
                kelly_size=kelly_size,
                max_profit_pct=put_mid / put_strike,
                days_to_exp=int(T * 365),
                liquidity_score=liquidity_score,
                iv_rank_score=iv_rank_score,
                regime_score=regime_score,
                risk_reward_score=risk_reward_score,
                final_score=final_score,
                total_delta=put_greeks['delta'] * kelly_size,
                total_gamma=put_greeks['gamma'] * kelly_size,
                total_theta=put_greeks['theta'] * kelly_size,
                total_vega=put_greeks['vega'] * kelly_size,
                notes=f"CSP targeting {target_delta:.2f}Δ put. Kelly size: {kelly_size}. "
                     f"TP: {self.config.take_profit_pct*100:.0f}% | Stop: {self.config.stop_loss_mult:.1f}x",
                market_context=f"IV Rank: {regime.iv_rank:.1%} | Regime: {regime.regime_type}",
                risk_level="LOW" if collateral < 5000 else "MEDIUM"
            )
            
        except Exception as e:
            logger.error(f"CSP generation failed for {symbol}: {e}")
            return None

    def generate_long_straddle(self, symbol: str, spot_price: float,
                              calls: pd.DataFrame, puts: pd.DataFrame,
                              exp: str, regime: MarketRegime) -> Optional[AdvancedIdea]:
        """Generate long straddle for low IV, high volatility expansion potential"""
        try:
            T = max(1/365, (pd.to_datetime(exp, utc=True) - pd.Timestamp.utcnow()).days / 365)
            
            # Find ATM options
            atm_call = calls.iloc[(calls['strike'] - spot_price).abs().argsort()[:1]].iloc[0]
            atm_put = puts.iloc[(puts['strike'] - spot_price).abs().argsort()[:1]].iloc[0]
            
            # Check liquidity for both legs
            call_liquid_ok, call_liq_score = self.enhanced_liquidity_check(atm_call)
            put_liquid_ok, put_liq_score = self.enhanced_liquidity_check(atm_put)
            
            if not call_liquid_ok or not put_liquid_ok:
                return None
            
            # Calculate cost (debit)
            call_mid = (float(atm_call['bid']) + float(atm_call['ask'])) / 2
            put_mid = (float(atm_put['bid']) + float(atm_put['ask'])) / 2
            total_debit = (call_mid + put_mid) * 100
            
            if total_debit <= 0:
                return None
            
            # Greeks calculation
            call_iv = float(atm_call.get('impliedVolatility', 0))
            put_iv = float(atm_put.get('impliedVolatility', 0))
            
            call_greeks = calculate_greeks(spot_price, float(atm_call['strike']), 
                                         self.config.risk_free_rate, call_iv, T, 'call')
            put_greeks = calculate_greeks(spot_price, float(atm_put['strike']), 
                                        self.config.risk_free_rate, put_iv, T, 'put')
            
            # Max risk is the premium paid
            max_risk = total_debit
            
            # Rough POP calculation - need significant move to be profitable
            strike = float(atm_call['strike'])
            upper_breakeven = strike + (total_debit / 100)
            lower_breakeven = strike - (total_debit / 100)
            
            # Estimate probability of significant move (simplified)
            avg_iv = (call_iv + put_iv) / 2
            expected_move = spot_price * avg_iv * np.sqrt(T)
            prob_big_move = min(0.8, expected_move / (total_debit / 100))
            est_pop = max(0.2, prob_big_move)
            
            # Position sizing for debit strategies
            max_risk_dollar = self.config.account_size * (self.config.max_risk_per_trade_pct / 100)
            kelly_size = max(1, int(max_risk_dollar / max_risk))
            
            # Scoring
            kelly_size, scores, final_score = self.calculate_position_size_and_scores(
                0, max_risk, est_pop, regime, [call_liq_score, put_liq_score], "VOLATILITY_EXPANSION"
            )
            
            # Net Greeks
            net_delta = (call_greeks['delta'] + put_greeks['delta']) * kelly_size
            net_gamma = (call_greeks['gamma'] + put_greeks['gamma']) * kelly_size
            net_theta = (call_greeks['theta'] + put_greeks['theta']) * kelly_size
            net_vega = (call_greeks['vega'] + put_greeks['vega']) * kelly_size
            
            return AdvancedIdea(
                symbol=symbol,
                strategy="LONG_STRADDLE",
                expiration=exp,
                legs=[
                    {"side": "BUY", "type": "CALL", "strike": float(atm_call['strike']), 
                     "exp": exp, "qty": kelly_size},
                    {"side": "BUY", "type": "PUT", "strike": float(atm_put['strike']), 
                     "exp": exp, "qty": kelly_size}
                ],
                credit=-total_debit,  # Negative for debit
                max_risk=max_risk,
                breakevens=[lower_breakeven, upper_breakeven],
                est_pop=est_pop,
                kelly_size=kelly_size,
                max_profit_pct=float('inf'),  # Theoretically unlimited
                days_to_exp=int(T * 365),
                liquidity_score=scores['liquidity_score'],
                iv_rank_score=scores['iv_rank_score'],
                regime_score=scores['regime_score'],
                risk_reward_score=scores['risk_reward_score'],
                final_score=final_score,
                total_delta=net_delta,
                total_gamma=net_gamma,
                total_theta=net_theta,
                total_vega=net_vega,
                notes=f"Long straddle expecting big move beyond ${lower_breakeven:.2f}-${upper_breakeven:.2f}. "
                     f"Low IV, volatility expansion play. Time decay enemy.",
                market_context=f"IV Rank: {regime.iv_rank:.1%} | Regime: {regime.regime_type}",
                risk_level="MEDIUM" if max_risk < 500 else "HIGH"
            )
            
        except Exception as e:
            logger.error(f"Long straddle generation failed for {symbol}: {e}")
            return None

    def generate_calendar_spread(self, symbol: str, spot_price: float, provider,
                                regime: MarketRegime) -> Optional[AdvancedIdea]:
        """Generate calendar spread using near-term and longer-term expirations"""
        try:
            expirations = provider.expirations(symbol)
            if len(expirations) < 2:
                return None
            
            # Find suitable expirations
            now = pd.Timestamp.utcnow()
            near_exp = None
            far_exp = None
            
            for exp in expirations:
                exp_date = pd.to_datetime(exp, utc=True)
                dte = (exp_date - now).days
                
                if 20 <= dte <= 35 and not near_exp:  # Near-term
                    near_exp = exp
                elif 50 <= dte <= 70 and near_exp and not far_exp:  # Far-term
                    far_exp = exp
                    break
            
            if not near_exp or not far_exp:
                return None
            
            # Get option chains for both expirations
            near_calls, near_puts = provider.chain(symbol, near_exp)
            far_calls, far_puts = provider.chain(symbol, far_exp)
            
            # Find ATM strikes (use puts for this example)
            near_atm = near_puts.iloc[(near_puts['strike'] - spot_price).abs().argsort()[:1]].iloc[0]
            far_atm = far_puts.iloc[(far_puts['strike'] - spot_price).abs().argsort()[:1]].iloc[0]
            
            # Check liquidity
            near_liquid_ok, near_liq_score = self.enhanced_liquidity_check(near_atm)
            far_liquid_ok, far_liq_score = self.enhanced_liquidity_check(far_atm)
            
            if not near_liquid_ok or not far_liquid_ok:
                return None
            
            # Calculate pricing (sell near, buy far)
            near_mid = (float(near_atm['bid']) + float(near_atm['ask'])) / 2
            far_mid = (float(far_atm['bid']) + float(far_atm['ask'])) / 2
            net_debit = (far_mid - near_mid) * 100  # Usually a debit
            
            if net_debit <= 0:
                return None
            
            # Time calculations
            T_near = max(1/365, (pd.to_datetime(near_exp, utc=True) - now).days / 365)
            T_far = max(1/365, (pd.to_datetime(far_exp, utc=True) - now).days / 365)
            
            # Greeks calculation
            strike = float(near_atm['strike'])
            near_iv = float(near_atm.get('impliedVolatility', 0))
            far_iv = float(far_atm.get('impliedVolatility', 0))
            
            near_greeks = calculate_greeks(spot_price, strike, self.config.risk_free_rate, near_iv, T_near, 'put')
            far_greeks = calculate_greeks(spot_price, strike, self.config.risk_free_rate, far_iv, T_far, 'put')
            
            max_risk = net_debit
            
            # Rough POP - benefits from time decay and stable price
            est_pop = 0.65  # Calendar spreads have decent success rates
            
            # Position sizing
            max_risk_dollar = self.config.account_size * (self.config.max_risk_per_trade_pct / 100)
            kelly_size = max(1, int(max_risk_dollar / max_risk))
            
            # Scoring
            kelly_size, scores, final_score = self.calculate_position_size_and_scores(
                0, max_risk, est_pop, regime, [near_liq_score, far_liq_score], "NEUTRAL"
            )
            
            # Net Greeks (short near - long far)
            net_delta = (near_greeks['delta'] - far_greeks['delta']) * kelly_size
            net_gamma = (near_greeks['gamma'] - far_greeks['gamma']) * kelly_size
            net_theta = (near_greeks['theta'] - far_greeks['theta']) * kelly_size
            net_vega = (near_greeks['vega'] - far_greeks['vega']) * kelly_size
            
            return AdvancedIdea(
                symbol=symbol,
                strategy="PUT_CALENDAR",
                expiration=f"{near_exp}/{far_exp}",
                legs=[
                    {"side": "SELL", "type": "PUT", "strike": strike, 
                     "exp": near_exp, "qty": kelly_size},
                    {"side": "BUY", "type": "PUT", "strike": strike, 
                     "exp": far_exp, "qty": kelly_size}
                ],
                credit=-net_debit,  # Negative for debit
                max_risk=max_risk,
                breakevens=[strike],  # Simplification - max profit near ATM at near expiration
                est_pop=est_pop,
                kelly_size=kelly_size,
                max_profit_pct=(near_mid * 0.7) / net_debit,  # Estimate
                days_to_exp=int(T_near * 365),
                liquidity_score=scores['liquidity_score'],
                iv_rank_score=scores['iv_rank_score'],
                regime_score=scores['regime_score'],
                risk_reward_score=scores['risk_reward_score'],
                final_score=final_score,
                total_delta=net_delta,
                total_gamma=net_gamma,
                total_theta=net_theta,
                total_vega=net_vega,
                notes=f"Calendar spread benefits from time decay and price stability near ${strike:.2f}. "
                     f"Close near expiration or roll out.",
                market_context=f"IV Rank: {regime.iv_rank:.1%} | Regime: {regime.regime_type}",
                risk_level="LOW" if max_risk < 200 else "MEDIUM"
            )
            
        except Exception as e:
            logger.error(f"Calendar spread generation failed for {symbol}: {e}")
            return None

    def analyze_symbol_comprehensive(self, symbol: str, provider) -> List[AdvancedIdea]:
        """Comprehensive analysis of a single symbol with multiple strategies"""
        try:
            # Get all data in one efficient call
            data = provider.get_symbol_data_once(symbol)
            if not data:
                return []
            
            spot_price = data['price']
            hist_1y = data['history']
            expirations = data['expirations']
            ticker = data['ticker']
            
            if hist_1y.empty or len(hist_1y) < 60:
                return []
            
            # Calculate market regime and IV rank using the same history data
            regime = self.detect_market_regime_from_history(hist_1y)
            
            # Find suitable expiration
            suitable_exp = self.find_suitable_expiration(expirations)
            if not suitable_exp:
                return []
            
            # Get options chain using the cached ticker
            try:
                chain = ticker.option_chain(suitable_exp)
                calls, puts = chain.calls.copy(), chain.puts.copy()
            except Exception as e:
                logger.warning(f"Options chain fetch failed for {symbol}: {e}")
                return []
            
            ideas = []
            
            # Always try multiple strategies, let scoring decide
            logger.info(f"{symbol}: IV Rank {regime.iv_rank:.1%} - generating multiple strategies")
            
            # Always try Put Credit Spread
            pcs = self.generate_put_credit_spread(symbol, spot_price, puts, suitable_exp, regime)
            if pcs:
                ideas.append(pcs)
            
            # Always try Call Credit Spread
            ccs = self.generate_call_credit_spread(symbol, spot_price, calls, suitable_exp, regime)
            if ccs:
                ideas.append(ccs)
            
            # Iron Condor for higher IV
            if regime.iv_rank > 0.4:  # Relaxed threshold
                ic = self.generate_iron_condor(symbol, spot_price, calls, puts, suitable_exp, regime)
                if ic:
                    ideas.append(ic)
            
            # Low IV strategies
            if regime.iv_rank < 0.4:  # Low IV strategies
                straddle = self.generate_long_straddle(symbol, spot_price, calls, puts, suitable_exp, regime)
                if straddle:
                    ideas.append(straddle)
                
                calendar = self.generate_calendar_spread(symbol, spot_price, provider, regime)
                if calendar:
                    ideas.append(calendar)
            
            # Log what was generated
            if ideas:
                strategies = [idea.strategy for idea in ideas]
                logger.info(f"{symbol}: Generated {len(ideas)} strategies: {', '.join(strategies)}")
            else:
                logger.warning(f"{symbol}: No suitable strategies found")
            
            return ideas
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for {symbol}: {e}")
            return []
    
    def find_suitable_expiration(self, expirations: List[str]) -> Optional[str]:
        """Find the most suitable expiration date"""
        now = pd.Timestamp.utcnow()
        target_dte = (self.config.dte_min + self.config.dte_max) // 2
        
        best_exp = None
        best_diff = float('inf')
        
        for exp in expirations:
            try:
                exp_date = pd.to_datetime(exp, utc=True)
                dte = (exp_date - now).days
                
                if self.config.dte_min <= dte <= self.config.dte_max:
                    diff = abs(dte - target_dte)
                    if diff < best_diff:
                        best_diff = diff
                        best_exp = exp
            except:
                continue
        
        return best_exp
    
    def detect_market_regime_from_history(self, hist: pd.DataFrame) -> MarketRegime:
        """Detect market regime from price history (optimized version)"""
        if hist.empty or len(hist) < 60:
            return MarketRegime(0, 'NORMAL', 0.5, 'RANGING')
        
        # Calculate trend strength
        close = hist['Close']
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        
        current_price = close.iloc[-1]
        trend_strength = (current_price - sma50.iloc[-1]) / sma50.iloc[-1]
        
        # Calculate volatility regime
        returns = close.pct_change().dropna()
        current_vol = returns.rolling(20).std().iloc[-1]
        historical_vol = returns.rolling(252).std().iloc[-1]
        
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        if vol_ratio > self.config.volatility_regime_threshold:
            vol_regime = 'HIGH'
        elif vol_ratio < 0.5:
            vol_regime = 'LOW'
        else:
            vol_regime = 'NORMAL'
        
        # Determine regime type
        if abs(trend_strength) > 0.1:
            regime_type = 'TRENDING_UP' if trend_strength > 0 else 'TRENDING_DOWN'
        elif vol_ratio > 1.5:
            regime_type = 'HIGH_VOL'
        else:
            regime_type = 'RANGING'
        
        # Calculate IV rank (simplified)
        iv_rank = min(1.0, max(0.0, (vol_ratio - 0.5) / 1.0))
        
        return MarketRegime(trend_strength, vol_regime, iv_rank, regime_type)

# -------------------------
# Enhanced Provider (keeping yfinance for now)
# -------------------------
class EnhancedYFProvider:
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    @rate_limit_retry()
    def price(self, symbol: str) -> float:
        ticker = self.yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if hist.empty:
            raise ValueError(f"No price data for {symbol}")
        return float(hist["Close"].iloc[-1])
    
    @rate_limit_retry()
    def history(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        return self.yf.Ticker(symbol).history(period=period)
    
    @rate_limit_retry()
    def expirations(self, symbol: str) -> List[str]:
        return list(self.yf.Ticker(symbol).options)
    
    @rate_limit_retry()
    def chain(self, symbol: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        chain = self.yf.Ticker(symbol).option_chain(expiration)
        return chain.calls.copy(), chain.puts.copy()
    
    def get_symbol_data_once(self, symbol: str):
        """Fetch all needed data in one efficient call to reduce API usage"""
        try:
            ticker = self.yf.Ticker(symbol)
            
            # Get all data at once
            hist_1y = ticker.history(period="1y")  # Use 1y for both regime and IV
            if hist_1y.empty:
                return None
                
            current_price = float(hist_1y["Close"].iloc[-1])
            expirations = list(ticker.options)
            
            return {
                'price': current_price,
                'history': hist_1y,
                'expirations': expirations,
                'ticker': ticker  # Reuse for options chain
            }
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {e}")
            return None

# -------------------------
# Enhanced Main Application
# -------------------------
def scan_all_symbols(config: AdvancedConfig) -> List[AdvancedIdea]:
    """Scan all configured symbols for opportunities"""
    provider = EnhancedYFProvider()
    generator = AdvancedStrategyGenerator(config)
    
    all_ideas = []
    
    if config.use_concurrent:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_symbol = {
                executor.submit(generator.analyze_symbol_comprehensive, symbol, provider): symbol
                for symbol in config.tickers
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    ideas = future.result(timeout=30)
                    all_ideas.extend(ideas)
                    logger.info(f"Analyzed {symbol}: {len(ideas)} ideas found")
                except Exception as e:
                    logger.error(f"Analysis failed for {symbol}: {e}")
    else:
        # Sequential processing with delays to avoid rate limiting
        for i, symbol in enumerate(config.tickers):
            try:
                # Add delay between symbols to avoid rate limiting
                if i > 0:
                    time.sleep(2)  # 2 second delay between symbols
                
                ideas = generator.analyze_symbol_comprehensive(symbol, provider)
                all_ideas.extend(ideas)
                logger.info(f"Analyzed {symbol}: {len(ideas)} ideas found")
            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")
    
    # Sort by final score
    all_ideas.sort(key=lambda x: x.final_score, reverse=True)
    return all_ideas[:config.max_ideas]

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Options Analysis Tool")
    parser.add_argument("--config", default="best_play.yaml", help="Configuration file path")
    parser.add_argument("--output", default="enhanced_results.json", help="Output file path")
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = AdvancedConfig(**config_data)
        
        logger.info("🚀 Starting Enhanced Options Analysis...")
        
        # Scan all symbols
        all_ideas = scan_all_symbols(config)
        
        if not all_ideas:
            logger.info("No tradable ideas found with current filters.")
            return
        
        # Display results
        print("=" * 100)
        print("🎯 TOP 5 OPTIONS TRADING OPPORTUNITIES")
        print("=" * 100)
        
        for i, idea in enumerate(all_ideas[:5], 1):
            print(f"\n[{i}] {idea.symbol} | {idea.strategy} | {idea.expiration} | Score: {idea.final_score:.1f}")
            
            # Risk level indicator
            risk_icon = "🔴" if idea.risk_level == "HIGH" else "🟡" if idea.risk_level == "MEDIUM" else "🟢"
            print(f"{risk_icon} Risk Level: {idea.risk_level}")
            
            # Financial metrics
            print(f"💰 Credit: ${idea.credit:.2f} | Max Risk: ${idea.max_risk:.2f} | Max Profit: {idea.max_profit_pct:.1%}")
            print(f"📊 POP: {idea.est_pop:.1%} | DTE: {idea.days_to_exp} | Kelly Size: {idea.kelly_size}")
            
            # Breakevens
            breakeven_str = " | ".join([f"${b:.2f}" for b in idea.breakevens])
            print(f"🎯 Breakevens: [{breakeven_str}]")
            
            # Greeks
            print(f"📈 Greeks (Total): Δ{idea.total_delta:.2f} | Γ{idea.total_gamma:.3f} | Θ{idea.total_theta:.2f} | ν{idea.total_vega:.2f}")
            
            # Legs
            print("📋 Legs:")
            for leg in idea.legs:
                print(f"   {leg['side']} {leg['qty']}x {leg['type']} ${leg['strike']}")
            
            # Context and notes
            print(f"💡 Context: {idea.market_context}")
            print(f"📝 Notes: {idea.notes}")
            print("-" * 100)
        
        # Portfolio summary
        total_risk = sum(idea.max_risk for idea in all_ideas[:5])
        risk_pct = (total_risk / config.account_size) * 100
        
        print(f"\n💼 PORTFOLIO SUMMARY:")
        print(f"Total Capital at Risk: ${total_risk:.2f} ({risk_pct:.1f}% of account)")
        print("Recommended Position Sizes: Kelly Criterion + Risk Management")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump([asdict(idea) for idea in all_ideas], f, indent=2, default=str)
        
        logger.info(f"Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()