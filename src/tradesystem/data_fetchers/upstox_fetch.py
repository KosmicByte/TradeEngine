import requests

url = 'https://api.upstox.com/v2/user/profile'
headers = {
    'Accept': 'application/json',
    'Authorization': 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzUEEyQkwiLCJqdGkiOiI2OGNiMGY0NDRhMzhkZjA2MmQwZDBhZWMiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzU4MTM4MTgwLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NTgxNDY0MDB9.Y3NPsAZgorHXX-4QHy4k75kv4CuuSwzV772dpnalX9U'
}
response = requests.get(url, headers=headers)

print(response.status_code)
print(response.json())