from gesture_fsm import GestureRecognizerFSM


def main() -> None:
    # Instantiating the FSM
    gesture_system = GestureRecognizerFSM()
    gesture_system.run()


if __name__ == "__main__":
    main()
