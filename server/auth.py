import json

command_permissions = {
  "hello": {"admin", "developer", "user"},
  "help": {"admin", "developer", "user"},
  "logs": {"admin", "developer"},
  "run_ml": {"admin", "developer", "user"}
}

with open('./database/user_data.jsonl', 'r') as f:
  data = json.load(f)

def authenticate(username, password):
  user = data.get(username)

  if not user:
    return False, "User doesn't exist"
  
  if user['password'] != password:
    return False, "Wrong Password"
  
  return True, None

def get_roles(username):
  user = data.get(username)
  return user.get('role') if user else None

def is_authorized(username, command):
  user_role = get_roles(username)
  
  if not user_role:
    return False, f"You have no role assigned"

  allowed_roles = command_permissions.get(command)
  
  if user_role not in allowed_roles:
    return False, f"You are not authorized to execute '{command}'"

  return True, None