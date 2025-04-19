import json
from threading import Lock
from collections import defaultdict

# Command handlers (Whenever we add a new command, add it into the Command registry and provide its proper function along with it)

def cmd_logs():
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

def cmd_help():
  help_output = "Available commands:\n"

  for cmd, handler in commands.items():
    desc = handler.__doc__ or "No description provided"
    help_output += f" - {cmd}: {desc}\n"

  return help_output.strip()

def cmd_hello():
  """Say hello to the server, what will it respond with???"""
  return "*Server decided to ignore you*"

def cmd_runAdvML(args=None):
    """Run AdvAttackDefence.py"""
    try:
        command = ["python", "-u", "AdvAttackDefence.py"]
        if args: 
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
  'run_ml': lambda args=None: cmd_runAdvML(args),
  'logs': cmd_logs
}