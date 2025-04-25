import json
import os
import subprocess
from threading import Lock
from collections import defaultdict
from hmac_check import generate_hmac

# Command handlers (Whenever we add a new command, add it into the Command registry and provide its proper function along with it)

def cmd_hmac(args=None):
  """Generates HMAC for given models"""
  model_path = './models/resnet18_traffic_signs.pth'

  if not os.path.isfile(model_path):
    return f"Error: File '{model_path}' couldn't be found"
  
  try:
    hmac_value = generate_hmac(model_path)
    hmac_path = model_path + ".hmac"

    with open(hmac_path, 'w') as f:
      f.write(hmac_value)

    return f"HMAC generated and saved to {hmac_path}\n\t-HMAC: {hmac_value}"
  except Exception as e:
    return f"Error generating HMAC: {str(e)}"

def cmd_logs(args=None):
  """Return logs for the current user"""
  logging_file = './logs/logs.jsonl'
  file_lock = Lock()

  full_logs = defaultdict(list)

  with file_lock:
    try:
      with open(logging_file, 'r') as f:
        for line in f:
          try:
            entry = json.loads(line.strip())
            user = entry.get("user", "unknown")
            full_logs[user].append(entry)

          except json.JSONDecodeError:
            continue
    except FileNotFoundError:
      return {}
    
  prettier_logs = ["Audit Logs:"]

  for user in sorted(full_logs.keys()):
    prettier_logs.append(f"\nUser: {user}")
    user_logs = sorted(full_logs[user], key=lambda e: e.get("timestamp", ""))
    
    for entry in user_logs:
      timestamp = entry.get("timestamp", "??")
      cmd = entry.get("command", "??")
      status = entry.get("status", "??")
      server_response = entry.get("server_response", "").replace('\n', ' ')

      prettier_logs.append(f"  [{timestamp}] {cmd} ({status}) -> {server_response}")

  return "\n".join(prettier_logs)

def cmd_help(args=None):
  """Showcases useful commands"""
  help_output = "Available commands:\n"

  for cmd, handler in commands.items():
    desc = handler.__doc__ or "No description provided"
    help_output += f" - {cmd}: {desc}\n"

  return help_output.strip()

def cmd_hello(args=None):
  """Say hello to the server, what will it respond with???"""
  return "*Server decided to ignore you*"

def cmd_runAdvML(args=None):
    """runs AdvAttackDefence.py 
       run_ml [--model {base, adv_trained}] [--train] [--attack {fgsm, pgd, deepfool, all}] [--defense {bitdepth, binary, jpeg, none, all}]
          --model	(Optional) Choose model type. Options: base, or adv_trained. Default is the base model.
          --train	(Optional) If set, the script trains a new base model. Otherwise, it loads a pre-trained model.
          --attack	(Optional) Choose attack method. Options: fgsm, pgd, deepfool, or all. Default is all.
          --defense	(Optional) Choose defense technique. Options: bitdepth, binary, jpeg, none, or all. Default is all.
"""
    try:
        args = args or []
        command = ["python", "-u", "AdvAttackDefence.py"]
        command += args

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Yield output line by line
        for line in process.stdout:
            yield line

        process.stdout.close()
        process.wait()

    except Exception as e:
        yield f"Error running file: {e}"
    
# Command registry
commands = {
  'help': cmd_help,
  'hello': cmd_hello,
  'run_ml': cmd_runAdvML,
  'logs': cmd_logs,
  'hmac': cmd_hmac
}