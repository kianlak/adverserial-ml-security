# ML Security Project - Server & Client Setup Guide

---

## Environment Setup

⚠️ Important: You will need **two terminal windows** — one for the **server** and one for the **client**.  
Activate the virtual environment in **both** terminals.

### Activate Virtual Environment

Use the appropriate command based on your operating system:

- Windows (Command Prompt):
  devenv\Scripts\activate.bat

- Windows (PowerShell):
  .\devenv\Scripts\Activate.ps1

- macOS / Linux:
  source devenv/bin/activate

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

- Ensure the virtual environment (`devenv`) is already created.
- Install dependencies with:
  pip install -r requirements.txt
- The server must be running before starting the client.

---
