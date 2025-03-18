import socket
import json
from naoqi import ALProxy


# NAOqi widget configuration
IP = "192.168.1.53"
tts = ALProxy("ALAnimatedSpeech", IP, 9559)
configuration = {"bodyLanguageMode": "contextual"}

# Socket connection configuration
HOST = ""
PORT = 12345
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))

# Intial Acknowlegement of the port
server_socket.listen(5)
print("Server is listening on port: ", PORT)
client_socket, address = server_socket.accept()
print("Client accepted from: ", address)


class PepperFSM:
    """Class that implements all the properties and behaviours of pepper."""
    def __init__(self):
        self.current_gesture = ""

    def recieve_message(self, data):
        """Function to decompile the json packet that was recieved"""
        try:
            message_decoded = json.loads(data)
            print("Message Recieved: ", message_decoded)
        except Exception as ex:
            print("Failed with: ", ex)
            return
        return message_decoded

    def process_message(self, message):
        """Function to determine the gesture_category as detected by the gesture recognition system."""
        if message:
            gesture_category = message.get("gesture_category")
            if gesture_category and (gesture_category != self.current_gesture):
                self.current_gesture = gesture_category

                if self.current_gesture == "Open_Palm":
                    tts.say("\\style=joyful\\ Oh, Hi!!!. Nice to meet you, I am Dusty.", configuration)
                elif self.current_gesture == "Thumb_Up":
                    tts.say("\\style=joyful\\ I am good!!!, thanks for asking. How are you ?", configuration)
                elif self.current_gesture == "Thumb_Down":
                    tts.say("\\style=neutral\\ Are you not having a good time ?, How can I help ?", configuration)
                elif self.current_gesture == "Victory":
                    tts.say("\\style=joyful\\  Oh, cool. Can we take a selfie together then ?", configuration)
                elif self.current_gesture == "ILoveYou":
                    tts.say("\\style=joyful\\  Awwwwwwwwwwwwwwww, I love you tooooo.", configuration)
                elif self.current_gesture == "Closed_Fist":
                    tts.say("\\style=joyful\\  Dammmn, you have a strong grip right there!!!", configuration)
                elif self.current_gesture == "Pointing_Up":
                    tts.say("\\style=joyful\\  Yup, thats right. Aim for sky, stars and beyond always !!!", configuration)

            elif gesture_category and (gesture_category == self.current_gesture):
                return
            else:
                print("No gesture detected !!!!!")

    def run(self):
        """Mainloop for the FSM of Pepper."""

        while True:
            # Reads 1024 bytes every iteration
            data = client_socket.recv(1024)

            if not data:
                continue

            message_decoded = self.recieve_message(data)
            self.process_message(message_decoded)


# Instantiating the class for run
if __name__ == "__main__":
    pepper_fsm = PepperFSM()
    pepper_fsm.run()
