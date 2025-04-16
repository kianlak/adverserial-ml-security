import requests
from client_utility import handle_command

url = 'http://localhost:5000/run'

print("Type your command (Type 'exit' to quit):")

while True:
  command = handle_command()

  if command == 'exit':
    print("Exiting...")
    break

  if not command:
    continue

  try:
    response = requests.post(url, json={'command': command})
    
    if response.ok:
      print(f"[Server]: {response.json().get('response')}")
    else:
      print(f"[ERROR]: {response.json().get('response')}")
  except requests.exceptions.RequestException as e:
    print("Connection failure:", e)
