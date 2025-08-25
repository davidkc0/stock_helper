#!/usr/bin/env python3
"""
Options Analysis Utilities
==========================
Helper functions and quick analysis tools for the enhanced options analyzer.

Features:
- Quick IV rank checker
- Position size calculator
- Greeks calculator
- Market regime analyzer
- Portfolio risk checker

Usage:
    python options_utilities.py --check-iv AAPL
    python options_utilities.py --calc-size --risk 1000 --pop 0.75
    python options_utilities.py --regime-check SPY
"""

import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import json

class OptionsUtilities:
    def __init__(self):
        self.risk_free_rate = 0.045
    
    def check_iv_rank(self, symbol: str, lookback_days: int = 252) -> dict:
        """Quick IV rank check for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price and options
            hist = ticker.history(period="2d")
            current_price = float(hist['Close'].iloc[-1])
            
            expirations = ticker.options
            if not expirations:
                return {"error": "No options available"}
            
            # Get nearest expiration
            exp = expirations[0]
            chain = ticker.option_chain(exp)
            
            # Find ATM options
            calls = chain.calls
            puts = chain.puts
            
            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
            
            if atm_call.empty or atm_put.empty:
                return {"error": "No ATM options found"}
            
            current_iv = (float(atm_call.iloc[0]['impliedVolatility']) + 
                         float(atm_put.iloc[0]['impliedVolatility'])) / 2
            
            # Get historical volatility for IV rank approximation
            hist_1y = ticker.history(period="1y")
            returns = np.log(hist_1y['Close'] / hist_1y['Close'].shift(1)).dropna()
            
            if len(returns) < 50:
                return {"error": "Insufficient historical data"}
            
            # Calculate rolling HV
            rolling_hv = returns.rolling(20).std() * np.sqrt(252)
            rolling_hv = rolling_hv.dropna()
            
            hv_min, hv_max = rolling_hv.min(), rolling_hv.max()
            current_hv = rolling_hv.iloc[-1]
            
            # IV rank approximation
            if hv_max > hv_min:
                iv_rank = (current_iv - hv_min) / (hv_max - hv_min)
            else:
                iv_rank = 0.5
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "current_iv": current_iv,
                "current_hv": current_hv,
                "hv_min_1y": hv_min,
                "hv_max_1y": hv_max,
                "iv_rank_approx": max(0, min(1, iv_rank)),
                "recommendation": self._get_iv_recommendation(iv_rank),
                "expiration": exp
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _get_iv_recommendation(self, iv_rank: float) -> str:
        """Get trading recommendation based on IV rank"""
        if iv_rank > 0.7:
            return "HIGH IV - Consider selling premium (credit spreads, iron condors)"
        elif iv_rank > 0.5:
            return "MEDIUM-HIGH IV - Moderate premium selling opportunities"
        elif iv_rank > 0.3:
            return "MEDIUM IV - Neutral strategies or wait for better setup"
        else:
            return "LOW IV - Consider buying premium (long calls/puts) or avoid trading"
    
    def calculate_position_size(self, account_size: float, max_risk_pct: float, 
                              trade_risk: float, win_rate: float, 
                              avg_win: float, avg_loss: float) -> dict:
        """Calculate optimal position size using multiple methods"""
        
        # Basic risk management sizing
        max_risk_dollar = account_size * (max_risk_pct / 100)
        basic_size = int(max_risk_dollar / trade_risk) if trade_risk > 0 else 0
        
        # Kelly Criterion sizing
        if avg_loss > 0:
            b = avg_win / avg_loss  # Odds received on winning bet
            p = win_rate  # Probability of winning
            kelly_fraction = (b * p - (1 - p)) / b
            kelly_size = int(account_size * kelly_fraction / trade_risk) if kelly_fraction > 0 else 0
        else:
            kelly_size = 0
            kelly_fraction = 0
        
        # Conservative Kelly (25% of full Kelly)
        conservative_kelly = int(kelly_size * 0.25)
        
        # Fixed fractional sizing (1% rule)
        fixed_fractional = int(account_size * 0.01 / trade_risk) if trade_risk > 0 else 0
        
        return {
            "account_size": account_size,
            "max_risk_dollar": max_risk_dollar,
            "trade_risk": trade_risk,
            "basic_size": max(1, basic_size),
            "kelly_fraction": kelly_fraction,
            "kelly_size": max(1, kelly_size),
            "conservative_kelly": max(1, conservative_kelly),
            "fixed_fractional": max(1, fixed_fractional),
            "recommended": max(1, min(basic_size, conservative_kelly)),
            "notes": self._get_sizing_notes(kelly_fraction, win_rate)
        }
    
    def _get_sizing_notes(self, kelly_fraction: float, win_rate: float) -> str:
        """Provide notes on position sizing"""
        notes = []
        
        if kelly_fraction <= 0:
            notes.append("Negative expected value - avoid this trade")
        elif kelly_fraction < 0.05:
            notes.append("Very small edge - consider skipping")
        elif kelly_fraction > 0.2:
            notes.append("Large edge detected - use conservative Kelly")
        
        if win_rate < 0.6:
            notes.append("Low win rate - ensure R:R is favorable")
        elif win_rate > 0.8:
            notes.append("High win rate - good for credit strategies")
        
        return " | ".join(notes) if notes else "Standard position sizing applies"
    
    def analyze_market_regime(self, symbol: str) -> dict:
        """Analyze current market regime for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if len(hist) < 100:
                return {"error": "Insufficient data for regime analysis"}
            
            close = hist['Close']
            volume = hist['Volume']
            
            # Trend analysis
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_100 = close.rolling(100).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # Trend strength
            trend_signals = []
            trend_signals.append(1 if current_price > sma_20 else -1)
            trend_signals.append(1 if sma_20 > sma_50 else -1)
            trend_signals.append(1 if sma_50 > sma_100 else -1)
            trend_strength = sum(trend_signals) / len(trend_signals)
            
            # Volatility analysis
            returns = close.pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            avg_vol = returns.rolling(100).std().mean() * np.sqrt(252)
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Volume analysis
            avg_volume = volume.rolling(50).mean().iloc[-1]
            recent_volume = volume.rolling(5).mean().iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Regime classification
            if abs(trend_strength) < 0.3:
                regime = "RANGING"
            elif trend_strength > 0.3:
                regime = "TRENDING_UP"
            else:
                regime = "TRENDING_DOWN"
            
            if vol_ratio > 1.5:
                volatility_regime = "HIGH_VOLATILITY"
            elif vol_ratio < 0.7:
                volatility_regime = "LOW_VOLATILITY"
            else:
                volatility_regime = "NORMAL_VOLATILITY"
            
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "trend_regime": regime,
                "trend_strength": float(trend_strength),
                "volatility_regime": volatility_regime,
                "volatility_ratio": float(vol_ratio),
                "rsi": float(rsi),
                "volume_ratio": float(volume_ratio),
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "sma_100": float(sma_100),
                "strategy_recommendation": self._get_regime_strategy(regime, volatility_regime, rsi)
            }
            
        except Exception as e:
            return {"error": f"Regime analysis failed: {str(e)}"}
    
    def _get_regime_strategy(self, trend: str, volatility: str, rsi: float) -> str:
        """Get strategy recommendation based on market regime"""
        recommendations = []
        
        if trend == "TRENDING_UP":
            recommendations.append("Favor bullish strategies (CSP, bull spreads)")
        elif trend == "TRENDING_DOWN":
            recommendations.append("Favor bearish strategies (bear spreads, long puts)")
        else:
            recommendations.append("Favor neutral strategies (iron condors, strangles)")
        
        if volatility == "HIGH_VOLATILITY":
            recommendations.append("High vol: sell premium (credit spreads)")
        elif volatility == "LOW_VOLATILITY":
            recommendations.append("Low vol: buy premium or avoid trading")
        
        if rsi > 70:
            recommendations.append("Overbought: consider bearish bias")
        elif rsi < 30:
            recommendations.append("Oversold: consider bullish bias")
        
        return " | ".join(recommendations)
    
    def calculate_greeks(self, S: float, K: float, r: float, iv: float, 
                        T: float, option_type: str = 'call') -> dict:
        """Calculate Black-Scholes Greeks"""
        try:
            if S <= 0 or K <= 0 or iv <= 0 or T <= 0:
                return {"error": "Invalid input parameters"}
            
            d1 = (np.log(S/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
            d2 = d1 - iv*np.sqrt(T)
            
            # Price
            if option_type.lower() == 'call':
                price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                delta = norm.cdf(d1)
                theta = (-(S*norm.pdf(d1)*iv)/(2*np.sqrt(T)) - 
                        r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
            else:
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                delta = norm.cdf(d1) - 1
                theta = (-(S*norm.pdf(d1)*iv)/(2*np.sqrt(T)) + 
                        r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
            
            # Common Greeks
            gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            rho = K * T * np.exp(-r*T) * (norm.cdf(d2) if option_type.lower() == 'call' else norm.cdf(-d2)) / 100
            
            return {
                "underlying_price": S,
                "strike": K,
                "time_to_expiry": T,
                "implied_volatility": iv,
                "risk_free_rate": r,
                "option_type": option_type,
                "theoretical_price": float(price),
                "delta": float(delta),
                "gamma": float(gamma),
                "theta": float(theta),
                "vega": float(vega),
                "rho": float(rho)
            }
            
        except Exception as e:
            return {"error": f"Greeks calculation failed: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="Options Analysis Utilities")
    parser.add_argument("--check-iv", metavar="SYMBOL", help="Check IV rank for symbol")
    parser.add_argument("--calc-size", action="store_true", help="Calculate position size")
    parser.add_argument("--regime-check", metavar="SYMBOL", help="Analyze market regime")
    parser.add_argument("--greeks", action="store_true", help="Calculate option Greeks")
    
    # Position sizing parameters
    parser.add_argument("--account", type=float, default=42000, help="Account size")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Max risk percentage")
    parser.add_argument("--trade-risk", type=float, help="Risk per trade")
    parser.add_argument("--win-rate", type=float, help="Historical win rate (0-1)")
    parser.add_argument("--avg-win", type=float, help="Average winning trade")
    parser.add_argument("--avg-loss", type=float, help="Average losing trade")
    
    # Greeks parameters
    parser.add_argument("--spot", type=float, help="Underlying price")
    parser.add_argument("--strike", type=float, help="Strike price")
    parser.add_argument("--iv", type=float, help="Implied volatility (decimal)")
    parser.add_argument("--dte", type=int, help="Days to expiration")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    
    args = parser.parse_args()
    
    utils = OptionsUtilities()
    
    if args.check_iv:
        print("🔍 Checking IV Rank for", args.check_iv.upper())
        print("=" * 50)
        result = utils.check_iv_rank(args.check_iv.upper())
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"Symbol: {result['symbol']}")
            print(f"Current Price: ${result['current_price']:.2f}")
            print(f"Trend Regime: {result['trend_regime']}")
            print(f"Trend Strength: {result['trend_strength']:.2f}")
            print(f"Volatility Regime: {result['volatility_regime']}")
            print(f"Vol Ratio: {result['volatility_ratio']:.2f}")
            print(f"RSI: {result['rsi']:.1f}")
            print(f"Volume Ratio: {result['volume_ratio']:.2f}")
            print()
            print("Moving Averages:")
            print(f"  SMA 20: ${result['sma_20']:.2f}")
            print(f"  SMA 50: ${result['sma_50']:.2f}")
            print(f"  SMA 100: ${result['sma_100']:.2f}")
            print()
            print(f"📋 Strategy Recommendation:")
            print(f"   {result['strategy_recommendation']}")
    
    elif args.greeks:
        if not all([args.spot, args.strike, args.iv, args.dte]):
            print("❌ Greeks calculation requires: --spot, --strike, --iv, --dte")
            return
        
        print("🔢 Option Greeks Calculation")
        print("=" * 50)
        T = args.dte / 365.0  # Convert days to years
        result = utils.calculate_greeks(
            args.spot, args.strike, utils.risk_free_rate, 
            args.iv, T, args.option_type
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"Underlying: ${result['underlying_price']:.2f}")
            print(f"Strike: ${result['strike']:.2f}")
            print(f"Time to Expiry: {args.dte} days ({result['time_to_expiry']:.4f} years)")
            print(f"Implied Volatility: {result['implied_volatility']:.1%}")
            print(f"Option Type: {result['option_type'].upper()}")
            print()
            print("Option Price & Greeks:")
            print(f"  Theoretical Price: ${result['theoretical_price']:.2f}")
            print(f"  Delta: {result['delta']:.4f}")
            print(f"  Gamma: {result['gamma']:.4f}")
            print(f"  Theta: ${result['theta']:.2f} (per day)")
            print(f"  Vega: {result['vega']:.2f}")
            print(f"  Rho: {result['rho']:.2f}")
            print()
            print("Greeks Interpretation:")
            if abs(result['delta']) > 0.5:
                print(f"  🎯 High delta - significant directional exposure")
            else:
                print(f"  🎯 Low delta - limited directional exposure")
            
            if result['gamma'] > 0.1:
                print(f"  ⚡ High gamma - delta will change rapidly")
            
            if abs(result['theta']) > 0.05:
                print(f"  ⏰ Significant time decay - ${abs(result['theta']):.2f} per day")
            
            if abs(result['vega']) > 0.1:
                print(f"  📊 High vega - sensitive to volatility changes")
    
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("USAGE EXAMPLES:")
        print("="*60)
        print()
        print("1. Check IV Rank:")
        print("   python options_utilities.py --check-iv AAPL")
        print()
        print("2. Calculate Position Size:")
        print("   python options_utilities.py --calc-size \\")
        print("       --account 42000 --risk-pct 1.0 --trade-risk 300 \\")
        print("       --win-rate 0.75 --avg-win 150 --avg-loss 250")
        print()
        print("3. Analyze Market Regime:")
        print("   python options_utilities.py --regime-check SPY")
        print()
        print("4. Calculate Greeks:")
        print("   python options_utilities.py --greeks \\")
        print("       --spot 150 --strike 155 --iv 0.25 --dte 45 --option-type call")
        print()

if __name__ == "__main__":
    main()Current IV: {result['current_iv']:.1%}")
            print(f"Current HV (20d): {result['current_hv']:.1%}")
            print(f"IV Rank (approx): {result['iv_rank_approx']:.1%}")
            print(f"📈 {result['recommendation']}")
    
    elif args.calc_size:
        if not all([args.trade_risk, args.win_rate, args.avg_win, args.avg_loss]):
            print("❌ Position sizing requires: --trade-risk, --win-rate, --avg-win, --avg-loss")
            return
        
        print("💰 Position Size Calculation")
        print("=" * 50)
        result = utils.calculate_position_size(
            args.account, args.risk_pct, args.trade_risk, 
            args.win_rate, args.avg_win, args.avg_loss
        )
        
        print(f"Account Size: ${result['account_size']:,.2f}")
        print(f"Max Risk: ${result['max_risk_dollar']:,.2f}")
        print(f"Trade Risk: ${result['trade_risk']:,.2f}")
        print(f"Kelly Fraction: {result['kelly_fraction']:.3f}")
        print()
        print("Position Sizes:")
        print(f"  Basic (risk mgmt): {result['basic_size']} contracts")
        print(f"  Kelly optimal: {result['kelly_size']} contracts")
        print(f"  Conservative Kelly: {result['conservative_kelly']} contracts")
        print(f"  Fixed fractional: {result['fixed_fractional']} contracts")
        print(f"  🎯 RECOMMENDED: {result['recommended']} contracts")
        print()
        print(f"Notes: {result['notes']}")
    
    elif args.regime_check:
        print("📊 Market Regime Analysis for", args.regime_check.upper())
        print("=" * 50)
        result = utils.analyze_market_regime(args.regime_check.upper())
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"Symbol: {result['symbol']}")
            print(f"Current Price: ${result['current_price']:.2f}")
            