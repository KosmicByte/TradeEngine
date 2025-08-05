
import requests

# Replace with your bot's token and your chat_id
BOT_TOKEN = 'your_bot_token'
CHAT_ID = 'your_chat_id'

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, data=payload)
        return response.ok
    except Exception as e:
        print("Telegram Error:", e)
        return False

# Example
if __name__ == "__main__":
    send_telegram_alert("🚨 Test Alert: Signal for NIFTY 24700 CE Buy @ 105. SL: 85, Target: 155")
