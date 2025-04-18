# Command handlers (Whenever we add a new command, add it into the Command registry and provide its proper function along with it)
import subprocess


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
  'run_ml': lambda args=None: cmd_runAdvML(args)
}