from flask import Flask, request, jsonify
from server_utility import commands
from auth import authenticate, is_authorized
from logger import insert_log

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
	data = request.json
	username = data.get('username', '')
	password = data.get('password', '')

	success, message = authenticate(username, password)

	if not success:
		return jsonify({'response': message}), 403
	
	return jsonify({'response': f"Access Granted"})

@app.route('/run', methods=['POST'])
def handle_command():
	data = request.json
	command = data.get('command', '')
	user = data.get('user', '')
	
	print(f"[+] Received command: {command}")

	if command in commands:
		try:
			authorized, message = is_authorized(user, command)
			
			if not authorized:
				insert_log(user, command, 'unauthorized', message)
				return jsonify({'response': message}), 403
			

			result = commands[command]()
			insert_log(user, command, 'authorized', message)

			return jsonify({'response': result})
		except Exception as e:
			return jsonify({'response': f"Error executing '{command}': {str(e)}"}), 500
	else:
		return jsonify({'response': f"Unknown command '{command}'. Type 'help' for a list of commands."}), 400

# This function should be at the bottom of the server
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)
