import requests

url = 'http://localhost:5000/run'

print("Interactive CLI Client. Type your command (or 'exit' to quit):")

while True:
  command = input(">> ").strip()
  if command.lower() == 'exit':
    print("👋 Exiting client.")
    break

  if not command:
    continue

  try:
    response = requests.post(url, json={'command': command})
    if response.ok:
      print(f"[Server]: {response.json().get('response')}")
    else:
      print("[-] Server error:", response.text)
  except requests.exceptions.RequestException as e:
    print("[-] Connection error:", e)
