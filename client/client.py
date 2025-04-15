import requests

url = 'http://localhost:5000/run'

print("Type your command (Type 'exit' to quit):")

def command_processing():
  processed_command = input(">> ").strip().lower()
  return processed_command

while True:
  command = command_processing()

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
      print(response.text)
  except requests.exceptions.RequestException as e:
    print("Connection failure:", e)
