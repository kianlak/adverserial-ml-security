import requests
import threading

from client_utility import handle_command, waiting_animation

url = 'http://localhost:5000/run'

print("Type your command (Type 'exit' to quit):")

while True:
  full_command = handle_command()
  parts = full_command.strip().split()

  if full_command == 'exit':
    print("Exiting...")
    break

  if not full_command:
    continue

  command = parts[0]
  args = parts[1:]

  try:
    stop_event = threading.Event()
    t = threading.Thread(target = waiting_animation, args = (stop_event, "Waiting"))
    t.start()

    response = requests.post(url, json = {'command': command, 'args': args}, stream=True)

    stop_event.set()
    t.join()
    
    if response.ok:
      print("[Server]: (Streaming Output)")
      for line in response.iter_lines(decode_unicode=True):
        if line:
          print(line)
    else:
      print(f"[ERROR]: {response.text}")
  except requests.exceptions.RequestException as e:
    print("Connection failure:", e)
