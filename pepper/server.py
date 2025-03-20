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
        self.current_mode = ""

        # Message to acknowledge the start of the experiment
        tts.say("\\style=neutral\\ The experiment has started, I am ready to go!!!!!!")

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
        message_type = message.get("message_type") if message else None

        # Constructing the logic based on message type
        if message and message_type == "send_message":

            # Differentiating the sending of messages based on the states
            state = message.get("state")

            # Identifying the gestures in the recognition state for response
            if state and state == "recognition":
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
                    else:
                        text = "\\style=joyful\\ Cool, this is the new gesture " + str(gesture_category) + " right !!!!"
                        tts.say(text, configuration)
                else:
                    return
            # Reponse for the new gesture tracked
            elif state and state == "capture":
                gesture_number = message.get("gesture_number")
                if gesture_number:
                    text = "\\style=joyful\\ Just learnt a new gesture " + str(gesture_number) + ", thank you !!!!"
                    tts.say(text, configuration)
            # Response when no gesture is detected in the send_message type
            else:
                print("No gesture detected !!!!!")

        # Constructing the logic based on state transitions for each entry point
        elif message and message_type == "entry_point":

            # The current state for the entry point
            state = message.get("state")
            if state and (self.current_mode != state):
                self.current_mode = state

                if state == "recognition":
                    tts.say("\\style=neutral\\ In gesture recognition mode.")
                elif state == "tracking":
                    tts.say("\\style=neutral\\ In gesture tracking mode.")
                elif state == "capture":
                    tts.say("\\style=neutral\\ In gesture capturing mode.")
                elif state == "learning":
                    tts.say("\\style=joyful\\ Thank you for teaching me the new gestures, just a second while I learn them !!!!")
                elif state == "exit":
                    text = "\\style=neutral\\ Oh, that's the end of the experiment. It was a pleasure to meet you. Bye Now!!!"
                    tts.say(text, configuration)
                    return 1
            else:
                return

    def run(self):
        """Mainloop for the FSM of Pepper."""

        while True:
            # Reads 1024 bytes every iteration
            data = client_socket.recv(1024)

            if not data:
                continue

            message_decoded = self.recieve_message(data)
            flag = self.process_message(message_decoded)
            if flag:
                server_socket.close()
                break


# Instantiating the class for run
if __name__ == "__main__":
    pepper_fsm = PepperFSM()
    pepper_fsm.run()
