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

  cd server <br>
  python server.py

---

## Client Setup

In the client terminal, run:

  cd client <br>
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

## Client Available commands:
 - help: Showcases useful commands
 - hello: Say hello to the server, what will it respond with???
 - run_ml: runs AdvAttackDefence.py <br>
   run_ml [--model {base, adv_trained}] [--train] [--attack {fgsm, pgd, deepfool, all}] [--defense {bitdepth, binary, jpeg, none, all}] <br>
     --model       (Optional) Choose model type. Options: base, or adv_trained. Default is the base model.<br>
     --train       (Optional) If set, the script trains a new base model. Otherwise, it loads a pre-trained model.<br>
     --attack      (Optional) Choose attack method. Options: fgsm, pgd, deepfool, or all. Default is all.<br>
     --defense     (Optional) Choose defense technique. Options: bitdepth, binary, jpeg, none, or all. Default is all.
 - logs: Return logs for the current user
 - hmac: Generates HMAC for given models
