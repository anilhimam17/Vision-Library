# Vision Library 👀

A library that leverages advanced computer vision techniques to accurately detect and interpret human gestures in real-time, addressing one of the
key challenges in human-robot interaction from the perspective of intelligent robot assistants, creating a seamless communication interface between humans and robot assistants.

## Features

The vision library includes the following features:
- Off the shelf **Gesture Recognition** of 7 Pre-Trained gestures.
    - Open Palm
    - Closed Fist
    - Thumb Up
    - Thumb Down
    - Pointing Up
    - Victory
    - I Love You (Sign Language)
- **Custom Gesture Learning** in Real-Time.
- **Hybrid Recognition pipeline** that polls the inferences made by the Pre-Trained model and Custom Classifier.
- **Contextual Natural Language** responses for the detected gestures to enhance the interactions.
- Runs on **local LLM's** to minimize the latency.
- Tailored to the Pepper Humanoid-Robot to generate animated speech responses. Scope for mapping Choregraphe animations to Pre-Trained and Custom Gestures.

## Demo and Dissertation Report

- To have a look at the demo please click [here](https://drive.google.com/file/d/1h-v_JkG5j-MQ3RtXYw6h8rlQaTwnwVFS/view?usp=sharing)
- To have a look at the in-depth technical dissertation report please click [here](https://drive.google.com/file/d/1QdT9XZFPpBnfPUjKUaokwlsXQ75-Dt8R/view?usp=sharing)

## Getting Started

- To reduce the pain of install all the depandencies I have utilised UV to manage all the requirements corresponding to the project. You can install UV by accessing the docs and command for the same through the [link](https://docs.astral.sh/uv/getting-started/installation/).
- On instally UV and cloning the repository execute following commands from the root path of the directory.
    - ```bash uv sync --all-groups```
    - ```bash uv run main.py```
