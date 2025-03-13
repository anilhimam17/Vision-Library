from gesture_finte_state_machine import GestureRecognizerFSM


def main() -> None:
    # Instantiating the FSM
    gesture_system = GestureRecognizerFSM()
    gesture_system.run()


if __name__ == "__main__":
    main()
