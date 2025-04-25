# ML Security Project - Server & Client Setup Guide

---

## Environment Setup

⚠️ Important: You will need **two terminal windows** — one for the **server** and one for the **client**.  
Activate the virtual environment in **both** terminals.

## Dependencies and Requirements

Make sure to run (in the root directory):

- To make sure you have the virtual environment ready:
  python -m venv venv

- To make sure you have all project dependencies:
  pip install -r requirements.txt

### Activate Virtual Environment

Use the appropriate command based on your operating system:

- Windows (Command Prompt):
  venv\Scripts\activate.bat

- Windows (PowerShell with Scripts enabled):
  .\venv\Scripts\Activate.ps1

- macOS / Linux:
  source venv/bin/activate

---

## Server Setup

In the server terminal, run:

  cd server
  python server.py

---

## Client Setup

In the client terminal, run:

  cd client
  python client.py

---

## Notes

- Ensure the virtual environment (`venv`) is already created.
- Install dependencies with when venv is activated:
  pip install -r requirements.txt
- The server must be running before starting the client.
- Mainly tested on Windows with Python 3.13.x
- On the first run, make sure to login to admin and run hmac to ensure a hashed hmac value for the model, after that you can use any user role
- admin login is Username: admin Password: pass

---
