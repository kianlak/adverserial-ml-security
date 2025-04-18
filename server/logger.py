import json
import time
from threading import Lock

logging_file = './logs/logs.jsonl'
file_lock = Lock()

def insert_log(user, command, status, details = None):
  log_entry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "user": user,
    "command": command,
    "status": status,
    "server_response": details or ""
  }

  with file_lock:
    with open(logging_file, 'a') as f:
      f.write(json.dumps(log_entry) + '\n')
