import socket
import json


HOST = ""
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)
print(f"Server is listening on port: {PORT}")

while True:
    client_socket, address = server_socket.accept()
    print(f"Client accepted from: {address}")
    data = client_socket.recv(1024)

    if not data:
        break
    try:
        message = json.loads(data)
        print("Recieved:\n", message)
    except Exception as e:
        print("Error: ", e)

    response = json.dumps({"status": "received"})
    _ = client_socket.send(response.encode("utf-8"))
    client_socket.close()
