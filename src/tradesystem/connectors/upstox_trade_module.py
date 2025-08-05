
# Upstox Live Trading Integration (sample)
# Make sure to install: pip install upstox-python

from upstox_api.api import *
import datetime
import pandas as pd

# CONFIG - Replace with your actual details
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
REDIRECT_URI = "http://localhost"
ACCESS_TOKEN = "your_saved_access_token"

# Initialize session (only needed once to get token)
def login_flow():
    session = Session(API_KEY)
    session.set_redirect_uri(REDIRECT_URI)
    print("Visit this URL and paste code after login:")
    print(session.get_login_url())

    code = input("Paste code: ")
    session.set_api_secret(API_SECRET)
    access_token = session.retrieve_access_token(code)
    print("Access token:", access_token)
    return access_token

# Initialize API
def init_upstox():
    u = Upstox(API_KEY, ACCESS_TOKEN)
    u.get_master_contract('NSE_EQ')
    u.get_master_contract('NSE_INDEX')
    u.get_master_contract('NSE_FO')
    return u

# Place a market order (Buy CE/PE)
def place_option_trade(upstox, symbol, strike_price, option_type, expiry, qty=50):
    contract = upstox.get_instrument_by_symbol('NSE_FO', f"NIFTY{expiry}{strike_price}{option_type}")
    order = upstox.place_order(
        TransactionType.Buy,
        contract,
        qty,
        OrderType.Market,
        ProductType.Intraday,
        DurationType.DAY,
        price=0,
        trigger_price=None,
        disclosed_quantity=None,
        stop_loss=None,
        square_off=None,
        trailing_ticks=None,
        is_amo=False
    )
    print("✅ Order placed:", order)

# Example usage
if __name__ == "__main__":
    # access_token = login_flow()  # Run once and save
    u = init_upstox()
    # Example: Buy 24700 CE expiring on 06JUN
    place_option_trade(u, "NIFTY", "24700", "CE", "06JUN24")
