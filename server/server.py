import inspect
from flask import Flask, request, jsonify, Response, stream_with_context
from server_utility import commands
app = Flask(__name__)

@app.route('/run', methods=['POST'])
def handle_command():
	data = request.json
	command = data.get('command', '')
	args = data.get('args', [])
	
	print(f"[+] Received command: {command} {args}")
	
	# if command in commands:
	# 	try:
	# 		result = commands[command]()
	# 		return jsonify({'response': result})
	# 	except Exception as e:
	# 		return jsonify({'response': f"Error executing '{command}': {str(e)}"}), 500
	# else:
	# 	return jsonify({'response': f"Unknown command '{command}'. Type 'help' for a list of commands."}), 400

	if command in commands:
		try:
			cmd_func = commands[command]

        # Check if function accepts args
			if len(inspect.signature(cmd_func).parameters) == 0:
				result = cmd_func()
			else:
				result = cmd_func(args)

			return Response(
                stream_with_context(result), 
                mimetype='text/plain'
            )
		except Exception as e:
			return Response(f"Error executing '{command}': {str(e)}", mimetype='text/plain'), 500
	else:
		return Response(f"Unknown command '{command}'", mimetype='text/plain'), 400
    

# This function should be at the bottom of the server
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)
