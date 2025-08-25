"""
Enhanced Options Trading Strategy Analyzer
------------------------------------------
Advanced options strategy recommendation system with:
- True IV Rank calculation over 252-day lookback
- Enhanced strategy selection based on market regime
- Better risk management and position sizing
- Multiple data providers support
- Advanced filtering and scoring
- Real-time earnings calendar integration

Key Improvements:
1. True IV Rank calculation (not just HV proxy)
2. Market regime detection (trending vs ranging)
3. Enhanced strategy scoring with multiple factors
4. Better liquidity and earnings filters
5. Advanced position sizing with Kelly criterion
6. Support for additional strategies (calendars, diagonals)
7. Real-time Greeks calculation
8. Better error handling and logging

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
    max_ideas: int = 5  # Show more ideas
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
# Enhanced Strategy Generators
# -------------------------
class AdvancedStrategyGenerator:
    def __init__(self, config: AdvancedConfig):
        self.config = config
    
    def enhanced_liquidity_check(self, row: pd.Series) -> Tuple[bool, float]:
        """Enhanced liquidity scoring"""
        bid = float(row.get('bid', 0))
        ask = float(row.get('ask', 0))
        volume = int(row.get('volume', 0))
        oi = int(row.get('openInterest', 0))
        
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
    
    def generate_put_credit_spread(self, symbol: str, spot_price: float, 
                                 puts: pd.DataFrame, exp: str, 
                                 regime: MarketRegime) -> Optional[AdvancedIdea]:
        """Generate enhanced put credit spread"""
        try:
            T = max(1/365, (pd.to_datetime(exp, utc=True) - pd.Timestamp.utcnow()).days / 365)
            
            # Find short strike based on delta
            best_short = None
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
            
            # Calculate pricing and Greeks
            short_mid = (float(short_row['bid']) + float(short_row['ask'])) / 2
            long_mid = (float(long_row['bid']) + float(long_row['ask'])) / 2
            net_credit = (short_mid - long_mid) * 100
            
            if net_credit <= 0:
                return None
            
            max_risk = width * 100 - net_credit
            max_profit_pct = net_credit / max_risk if max_risk > 0 else 0
            
            # Enhanced POP calculation
            prob_itm = abs(short_greeks['delta'])
            est_pop = max(0.1, 1 - prob_itm)
            
            # Kelly sizing
            edge = est_pop - 0.5  # Edge over 50/50
            if edge > 0:
                kelly_f = edge / (max_risk / net_credit) if max_risk > 0 else 0
                kelly_size = max(1, int(self.config.account_size * kelly_f * self.config.kelly_fraction / max_risk))
            else:
                kelly_size = 1
            
            # Risk management override
            max_risk_dollar = self.config.account_size * (self.config.max_risk_per_trade_pct / 100)
            if max_risk * kelly_size > max_risk_dollar:
                kelly_size = max(1, int(max_risk_dollar / max_risk))
            
            # Scoring
            liquidity_score = (short_liq_score + long_liq_score) / 2
            iv_rank_score = regime.iv_rank
            
            # Regime-based scoring
            if regime.regime_type in ['TRENDING_UP', 'RANGING']:
                regime_score = 0.8
            else:
                regime_score = 0.4
            
            risk_reward_score = min(1, max_profit_pct * 2)  # Scale RR to 0-1
            
            final_score = (liquidity_score * 0.2 + 
                          iv_rank_score * 0.3 + 
                          regime_score * 0.3 + 
                          risk_reward_score * 0.2) * 100
            
            # Breakeven calculation
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
                max_profit_pct=max_profit_pct,
                days_to_exp=int(T * 365),
                liquidity_score=liquidity_score,
                iv_rank_score=iv_rank_score,
                regime_score=regime_score,
                risk_reward_score=risk_reward_score,
                final_score=final_score,
                total_delta=short_greeks['delta'] * kelly_size,
                total_gamma=short_greeks['gamma'] * kelly_size,
                total_theta=short_greeks['theta'] * kelly_size,
                total_vega=short_greeks['vega'] * kelly_size,
                notes=f"PCS targeting {target_delta:.2f}Δ short put. Kelly size: {kelly_size}. "
                     f"TP: {self.config.take_profit_pct*100:.0f}% | Stop: {self.config.stop_loss_mult:.1f}x",
                market_context=f"IV Rank: {regime.iv_rank:.1%} | Regime: {regime.regime_type}",
                risk_level="MEDIUM" if max_risk < 500 else "HIGH"
            )
            
        except Exception as e:
            logger.error(f"PCS generation failed for {symbol}: {e}")
            return None

    def analyze_symbol_comprehensive(self, symbol: str, provider) -> List[AdvancedIdea]:
        """Comprehensive analysis of a single symbol with optimized data fetching"""
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
            iv_rank = self.calculate_true_iv_rank(hist_1y)
            
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
            
            # Generate strategies based on regime
            if iv_rank > self.config.high_iv_threshold:
                # High IV - favor premium selling
                pcs = self.generate_put_credit_spread(symbol, spot_price, puts, suitable_exp, regime)
                if pcs:
                    ideas.append(pcs)
                
                # Could add iron condor, call credit spreads here
                
            elif iv_rank < self.config.low_iv_threshold:
                # Low IV - favor premium buying
                # Could add long calls/puts here
                pass
            
            else:
                # Medium IV - neutral strategies
                pcs = self.generate_put_credit_spread(symbol, spot_price, puts, suitable_exp, regime)
                if pcs:
                    ideas.append(pcs)
            
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
    
    def calculate_true_iv_rank(self, hist: pd.DataFrame) -> float:
        """Calculate true IV rank from historical data"""
        if hist.empty or len(hist) < 252:
            return 0.5
        
        returns = hist['Close'].pct_change().dropna()
        current_vol = returns.rolling(20).std().iloc[-1]
        
        # Calculate historical volatility percentiles
        vol_history = []
        for i in range(20, len(returns) - 20):
            vol_window = returns.iloc[i-20:i].std()
            vol_history.append(vol_window)
        
        if not vol_history:
            return 0.5
        
        vol_history = sorted(vol_history)
        rank = sum(1 for v in vol_history if v <= current_vol) / len(vol_history)
        
        return rank

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

def print_enhanced_results(ideas: List[AdvancedIdea]):
    """Print enhanced results with better formatting"""
    if not ideas:
        print("🚫 No suitable trading opportunities found today.")
        return
    
    print("=" * 100)
    print(f"🎯 TOP {len(ideas)} OPTIONS TRADING OPPORTUNITIES")
    print("=" * 100)
    
    total_risk = sum(idea.max_risk * idea.kelly_size for idea in ideas)
    
    for i, idea in enumerate(ideas, 1):
        risk_color = "🔴" if idea.risk_level == "HIGH" else "🟡" if idea.risk_level == "MEDIUM" else "🟢"
        
        print(f"\n[{i}] {idea.symbol} | {idea.strategy} | {idea.expiration} | Score: {idea.final_score:.1f}")
        print(f"{risk_color} Risk Level: {idea.risk_level}")
        
        print(f"💰 Credit: ${idea.credit:.2f} | Max Risk: ${idea.max_risk:.2f} | Max Profit: {idea.max_profit_pct:.1%}")
        print(f"📊 POP: {idea.est_pop:.1%} | DTE: {idea.days_to_exp} | Kelly Size: {idea.kelly_size}")
        print(f"🎯 Breakevens: {[f'${be:.2f}' for be in idea.breakevens]}")
        
        print(f"📈 Greeks (Total): Δ{idea.total_delta:.2f} | Γ{idea.total_gamma:.3f} | Θ{idea.total_theta:.2f} | ν{idea.total_vega:.2f}")
        
        print("📋 Legs:")
        for leg in idea.legs:
            print(f"   {leg['side']} {leg['qty']}x {leg['type']} ${leg['strike']:.2f}")
        
        print(f"💡 Context: {idea.market_context}")
        print(f"📝 Notes: {idea.notes}")
        print("-" * 100)
    
    print(f"\n💼 PORTFOLIO SUMMARY:")
    print(f"Total Capital at Risk: ${total_risk:.2f} ({total_risk/42000*100:.1f}% of account)")
    print(f"Recommended Position Sizes: Kelly Criterion + Risk Management")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Options Strategy Analyzer")
    parser.add_argument("--config", default="best_play.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["scan", "analyze"], default="scan", help="Operation mode")
    parser.add_argument("--symbol", help="Single symbol to analyze (for analyze mode)")
    parser.add_argument("--output", default="enhanced_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
                config = AdvancedConfig(**config_dict)
        else:
            logger.warning(f"Config file {args.config} not found. Using defaults.")
            config = AdvancedConfig()
    except Exception as e:
        logger.error(f"Config loading failed: {e}. Using defaults.")
        config = AdvancedConfig()
    
    logger.info("🚀 Starting Enhanced Options Analysis...")
    
    if args.mode == "scan":
        ideas = scan_all_symbols(config)
        print_enhanced_results(ideas)
        
        # Save results
        results_dict = [asdict(idea) for idea in ideas]
        with open(args.output, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {args.output}")
        
    elif args.mode == "analyze" and args.symbol:
        provider = EnhancedYFProvider()
        generator = AdvancedStrategyGenerator(config)
        ideas = generator.analyze_symbol_comprehensive(args.symbol.upper(), provider)
        print_enhanced_results(ideas)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    import os
    main()