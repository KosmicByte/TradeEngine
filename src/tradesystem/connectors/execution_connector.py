from upstox_client import UpstoxClient
from upstox_client.api import InstrumentApi, OrderApi
from upstox_client.models import (
    PlaceOrderRequest,
    TransactionType,
    OrderType,
    ProductType,
    DurationType
)
from upstox_client.rest import ApiException

class ExecutionManager:
    def __init__(self, access_token, capital):
        """
        access_token: Upstox access token (OAuth)
        capital: total trading capital (INR)
        """
        self.capital = capital

        # Setup client
        configuration = UpstoxClient.Configuration()
        configuration.access_token = access_token
        self.api_client = UpstoxClient(configuration)

        self.order_api = OrderApi(self.api_client)
        self.instrument_api = InstrumentApi(self.api_client)

    def calculate_kelly_lot(self, win_rate, rr_ratio, risk_per_trade=0.01, price_per_point=1):
        b = rr_ratio
        p = win_rate
        q = 1 - p
        k = (b * p - q) / b
        k = max(min(k, 1), 0)
        risk_amount = self.capital * risk_per_trade
        lot_qty = max(int((k * self.capital) / (price_per_point * b)), 1)
        return lot_qty

    def get_instrument_token(self, exchange, symbol):
        """
        Use Instrument API to fetch token for a given symbol.
        """
        instruments = self.instrument_api.get_instruments(exchange_segment=exchange)
        for ins in instruments:
            if ins.tradingsymbol == symbol:
                return ins.instrument_token
        raise ValueError(f"Symbol {symbol} not found in {exchange}")

    def place_option_order(self, symbol, strike, option_type, expiry, qty):
        """
        Place market order for options.
        """
        inst_symbol = f"{symbol}{expiry}{strike}{option_type}"  # e.g., NIFTY06JUN2517500CE

        try:
            token = self.get_instrument_token('NSE_FO', inst_symbol)

            order_req = PlaceOrderRequest(
                transaction_type=TransactionType.BUY if option_type == 'CE' else TransactionType.SELL,
                instrument_token=token,
                quantity=qty,
                order_type=OrderType.MARKET,
                product=ProductType.INTRADAY,
                duration=DurationType.DAY
            )
            result = self.order_api.place_order(order_req)
            return result
        except ApiException as e:
            print("API error:", e)
        except Exception as e:
            print("Order placement failed:", e)
        return None

    def execute_trade(self, symbol, strike, option_type, expiry, win_rate, rr_ratio, risk_per_trade=0.01):
        qty = self.calculate_kelly_lot(win_rate, rr_ratio, risk_per_trade)
        print(f"Calculated lot size: {qty}")
        order = self.place_option_order(symbol, strike, option_type, expiry, qty)
        if order:
            print("Order placed:", order)
        return order

# Usage:
# em = ExecutionManager(ACCESS_TOKEN, capital=100000)
# em.execute_trade("NIFTY", 17500, "CE", "06JUN25", win_rate=0.6, rr_ratio=2.0)