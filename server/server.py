from flask import Flask, request, jsonify
from server_utility import commands
app = Flask(__name__)

@app.route('/run', methods=['POST'])
def handle_command():
	data = request.json
	command = data.get('command', '')
	
	print(f"[+] Received command: {command}")
	
	if command in commands:
		try:
			result = commands[command]()
			return jsonify({'response': result})
		except Exception as e:
			return jsonify({'response': f"Error executing '{command}': {str(e)}"}), 500
	else:
		return jsonify({'response': f"Unknown command '{command}'. Type 'help' for a list of commands."}), 400

# This function should be at the bottom of the server
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)
