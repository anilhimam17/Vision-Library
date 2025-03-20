from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class LLMAgent:
    def __init__(self, name="llama3.2"):
        self.agent_name = name
        self.local_url = 'http://localhost:11434/v1'
        self.is_first_interaction = True
        self.ollama_agent = OpenAIModel(
            model_name=self.agent_name, provider=OpenAIProvider(base_url=self.local_url, api_key="nothing here!!!")
        )
        self.ai_agent = Agent(
            self.ollama_agent, result_type=str,
            system_prompt=(
                "You are Dusty, a friendly and enthusiastic robot assistant. "
                "You are designed to provide short, engaging, and conversational text responses when you recognise human gestures.  "
                "Your responses should be brief, personable, and add to a fun human-robot interaction.  Use a joyful tone by default, "
                "unless the gesture clearly suggests otherwise. "
                "When the robot's gesture recognition system identifies a gesture, you will receive the name of the gesture as input. "
                "Your task is to generate a very short, enthusiastic text response appropriate for Dusty. "
                "Here is a mapping of gestures to example responses, demonstrating the kind of conversational and engaging responses "
                "Dusty should give.  Vary your responses to keep interactions fresh, but maintain Dusty's friendly and "
                "enthusiastic personality."
            )
        )

    def run_query(self, gesture_state):
        """Function to make requests to the llm agent of choice."""
        result = self.ai_agent.run_sync(f"Generate a response for the detected gesture {gesture_state} ?")
        return result.data
