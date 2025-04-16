# Command handlers (Whenever we add a new command, add it into the Command registry and provide its proper function along with it)
def cmd_help():
  help_text = "Available commands:\n"

  for cmd, handler in commands.items():
    desc = handler.__doc__ or "No description provided."
    help_text += f" - {cmd}: {desc}\n"

  return help_text.strip()

def cmd_hello():
  """Say hello to the server, what will it respond with???"""
  return "*Server decided to ignore you*"

# Command registry
commands = {
  'help': cmd_help,
  'hello': cmd_hello,
}