import time
import sys

def handle_command():
  command = input(">> ").strip().lower()
  return command

def waiting_animation(stop_event, message = "Waiting"):
    dot_count = 0

    while not stop_event.is_set():
      dots = '.' * (dot_count % 4)

      sys.stdout.write(f'\r{message}{dots}{" " * (3 - len(dots))}')
      sys.stdout.flush()

      dot_count += 1

      time.sleep(0.25)

    sys.stdout.write('\r' + ' ' * (len(message) + 3) + '\r')