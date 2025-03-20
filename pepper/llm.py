from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class LLMAgent:
    def __init__(self, name="deepseek-r1:1.5b"):
        self.agent_name = name
        self.local_url = 'http://localhost:11434/v1'
        self.ollama_agent = OpenAIModel(
            model_name=self.agent_name, provider=OpenAIProvider(base_url=self.local_url, api_key="nothing here!!!")
        )
        self.ai_agent = Agent(
            self.ollama_agent, result_type=str,
            system_prompt=(
                "You are generating responses based on gesture states you will recieve for a consumer robot "
                "keep it brief, simple, polite, fun and engaging to improve the quality of the gesture recogntion "
                "interaction with the users."
            )
        )

    def run_query(self, gesture_state: str):
        result = self.ai_agent.run_sync(f"Detected: {gesture_state}, generate a response to the gesture ?")
        return result.data
