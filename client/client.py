import requests
import threading

from client_utility import handle_command, waiting_animation

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
    stop_event = threading.Event()
    t = threading.Thread(target = waiting_animation, args = (stop_event, "Waiting"))
    t.start()

    response = requests.post(url, json = {'command': command})

    stop_event.set()
    t.join()
    
    if response.ok:
      print(f"[Server]: {response.json().get('response')}")
    else:
      print(f"[ERROR]: {response.json().get('response')}")
  except requests.exceptions.RequestException as e:
    print("Connection failure:", e)
