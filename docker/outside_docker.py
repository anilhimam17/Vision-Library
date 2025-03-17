import socket
import json

HOST = 'localhost'  # Assuming you're connecting from the same machine
PORT = 2500         # Must match the host port mapped

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

# Send a JSON message
message = json.dumps({"key": "value"})
client_socket.send(message.encode("utf-8"))

response = client_socket.recv(1024)
print("Response from server:", json.loads(response))
client_socket.close()
