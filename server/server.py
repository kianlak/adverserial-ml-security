from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def handle_command():
    data = request.json
    command = data.get('command', '')
    print(f"[+] Received command: {command}")
    return jsonify({'response': 'hello'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
