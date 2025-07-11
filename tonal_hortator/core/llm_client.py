import ollama


class LocalLLMClient:
    """
    Lightweight wrapper around local Ollama for generating structured responses
    """

    def __init__(self, model_name: str = "llama3:8b"):
        self.model_name = model_name

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
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from LLM: {e}")
