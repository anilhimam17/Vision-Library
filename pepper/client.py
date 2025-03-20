import socket
import json


class PepperSocket:
    def __init__(self, host="192.168.1.53", port=12345):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self.client_socket.connect((self.host, self.port))

    def send_message(self, state, message=None, message_type=None):
        """Function to abstract the message sending mechanism to pepper."""
        json_packet = json.dumps({})

        if message_type == "send_message":
            if state == "recognition":
                json_packet = json.dumps({
                    "state": state,
                    "gesture_category": message,
                    "message_type": message_type
                })
            elif state == "capture":
                json_packet = json.dumps({
                    "state": state,
                    "gesture_number": message,
                    "message_type": message_type
                })
        elif message_type == "entry_point":
            if state == "recognition":
                json_packet = json.dumps({
                    "state": state,
                    "message_type": message_type
                })
            elif state == "tracking":
                json_packet = json.dumps({
                    "state": state,
                    "message_type": message_type
                })
            elif state == "capture":
                json_packet = json.dumps({
                    "state": state,
                    "message_type": message_type
                })
            elif state == "exit":
                json_packet = json.dumps({
                    "state": state,
                    "message_type": message_type
                })

        # Transmit the Message
        _ = self.client_socket.send(json_packet.encode("utf-8"))

    def close_socket(self):
        "Function to close the socket connection on termination."
        self.client_socket.close()
