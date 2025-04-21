import hmac
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

def generate_hmac(model):
  generated_hmac = hmac.new(SECRET_KEY.encode(), digestmod = hashlib.sha512)
  
  with open(model, 'rb') as f:
    for chunk in iter(lambda: f.read(10240), b""):
      generated_hmac.update(chunk)
  return generated_hmac.hexdigest()

def verifying_hmac(model, hmac_file_path):
  try:
    with open(hmac_file_path, 'r') as f:
      expected_hmac = f.read().strip()
  except Exception as e:
    return False, f"Error reading HMAC file: {str(e)}"

  try:
    generated = generate_hmac(model)
  except Exception as e:
    return False, f"Error generating HMAC: {str(e)}"

  if hmac.compare_digest(generated, expected_hmac):
    return True, "HMAC check passed."
  else:
    return False, "HMAC check failed."