import requests
import threading

from client_utility import handle_command, waiting_animation

url = 'http://localhost:5000'
current_user = None

while True:
  print("Please login to use:")
  username = input("Username: ").strip()
  password = input("Password: ").strip()

  try:
    stop_event = threading.Event()
    t = threading.Thread(target = waiting_animation, args = (stop_event, "Waiting"))
    t.start()

    response = requests.post(f'{url}/login', json={'username': username, 'password': password})
    
    stop_event.set()
    t.join()

    if response.ok:
      print(f"[Server]: {response.json().get('response')}")
      current_user = username
      break
    else:
      print(f"[ERROR]: {response.json().get('response')}")
  except requests.exceptions.RequestException as e:
    print("[FATAL ERROR]:", e)
    exit(1)

print("Type your command (Type 'exit' to quit):")

while True:
  command = handle_command()

  if command == 'exit':
    print("Exiting...")
    break

  if not command:
    continue

  try:
    stop_event = threading.Event()
    t = threading.Thread(target = waiting_animation, args = (stop_event, "Waiting"))
    t.start()

    response = requests.post(f'{url}/run', json={'command': command, 'user': current_user})

    stop_event.set()
    t.join()
    
    if response.ok:
      print(f"[Server]: {response.json().get('response')}")
    else:
      print(f"[ERROR]: {response.json().get('response')}")
  except requests.exceptions.RequestException as e:
    print("[FATAL ERROR]", e)
