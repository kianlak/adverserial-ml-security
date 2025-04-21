import time
from collections import defaultdict

MAX_REQUESTS = 3
COUNTDOWN = 60

users_request_count = defaultdict(list)

def check_rate_limited(user):
  current_time = time.time()
  countdown_start = current_time - COUNTDOWN

  users_request_count[user] = [t for t in users_request_count[user] if t >= countdown_start]

  if len(users_request_count[user]) >= MAX_REQUESTS:
    return True

  users_request_count[user].append(current_time)
  return False
