import logging

import ollama

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """
    Lightweight wrapper around local Ollama for generating structured responses
    """

    def __init__(
        self, model_name: str = "llama3:8b", prompt_path: str = "llm_prompt.txt"
    ):
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.load_prompt()

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a response from the local LLM using Ollama

        Args:
            prompt: The prompt string to send to the model
            max_tokens: Maximum tokens in the response

        Returns:
            Response text
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                options={"num_predict": max_tokens},
            )
            return response["message"]["content"]  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from LLM: {e}")

    def load_prompt(self) -> None:
        """Load prompt from file, handling case where file doesn't exist."""
        try:
            with open(self.prompt_path, "r") as f:
                self.prompt = f.read()
                logger.debug(f"Loaded prompt from {self.prompt_path}")
        except FileNotFoundError:
            self.prompt = ""
            logger.warning(
                f"Prompt file not found: {self.prompt_path}. Using empty prompt."
            )

    def reload_prompt(self) -> None:
        self.load_prompt()
        print("Prompt reloaded!")
