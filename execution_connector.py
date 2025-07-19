
import time
import numpy as np
from upstox_api.api import Upstox, TransactionType, OrderType, ProductType, DurationType

class ExecutionManager:
    def __init__(self, api_key, access_token, secret, capital):
        """
        api_key, access_token, secret: Upstox credentials
        capital: total trading capital (INR)
        """
        self.capital = capital
        self.u = Upstox(api_key, access_token)
        # Load contracts
        self.u.get_master_contract('NSE_FO')
        self.u.get_master_contract('NSE_EQ')
        self.u.get_master_contract('NSE_INDEX')

    def calculate_kelly_lot(self, win_rate, rr_ratio, risk_per_trade=0.01, price_per_point=1):
        """
        Calculate lot size using Kelly Criterion.
        win_rate: expected win probability (0-1)
        rr_ratio: reward:risk ratio (e.g., 2 for 1:2)
        risk_per_trade: fraction of capital to risk per trade
        price_per_point: value of one point per lot
        """
        b = rr_ratio
        p = win_rate
        q = 1 - p
        k = (b * p - q) / b
        k = max(min(k, 1), 0)  # bound between 0 and 1
        # risk amount = capital * risk_per_trade
        risk_amount = self.capital * risk_per_trade
        # points_at_risk = risk_amount / price_per_point
        # lot size = k * capital / (points_at_risk * lot_size_multiplier)
        # Upstox lot size is fixed for index (25 for NIFTY) or can be calculated for stocks.
        # Simplest: lot_qty = floor(k * (capital / (price_per_point * b)))
        lot_qty = max(int((k * self.capital) / (price_per_point * b)), 1)
        return lot_qty

    def place_option_order(self, symbol, strike, option_type, expiry, qty, stop_loss=None, target=None):
        """
        Place a market order with Upstox for CE/PE options.
        option_type: 'CE' or 'PE'
        expiry: in format 'DDMMMYY' e.g., '06JUN25'
        qty: number of lots
        stop_loss/target: triggers for SL/TP orders (not fully supported by Upstox API)
        """
        # Construct instrument symbol, e.g., 'NIFTY06JUN2517500CE'
        inst_symbol = f"{symbol}{expiry}{strike}{option_type}"
        try:
            contract = self.u.get_instrument_by_symbol('NSE_FO', inst_symbol)
            order = self.u.place_order(
                TransactionType.Buy if option_type == 'CE' else TransactionType.Sell,
                contract,
                qty,
                OrderType.Market,
                ProductType.Intraday,
                DurationType.DAY,
                price=0,
                trigger_price=None
            )
            return order
        except Exception as e:
            print("Order placement failed:", e)
            return None

    def execute_trade(self, symbol, strike, option_type, expiry, win_rate, rr_ratio, risk_per_trade=0.01):
        """
        High-level method:
        - Calculate lot size via Kelly
        - Place market order
        """
        qty = self.calculate_kelly_lot(win_rate, rr_ratio, risk_per_trade)
        print(f"Calculated lot size: {qty}")
        order = self.place_option_order(symbol, strike, option_type, expiry, qty)
        if order:
            print("Order placed:", order)
        return order

# Example usage:
# from execution_connector import ExecutionManager
# em = ExecutionManager(API_KEY, ACCESS_TOKEN, API_SECRET, capital=100000)
# em.execute_trade("NIFTY", 17500, "CE", "06JUN25", win_rate=0.6, rr_ratio=2.0)
