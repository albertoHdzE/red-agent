from pathlib import Path

from jinja2 import Template
from langchain_ollama import ChatOllama


class DebateAgent:
    def __init__(self, name: str, role: str, model: str = "mistral"):
        self.name = name
        self.role = role
        self.memory = []
        self.finished = False
        self.model_name = model
        self.turn_count = 0
        self.min_turns = 2  # Agents must contribute at least 2 times

        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts/debate_prompt.jinja"
        self.template = Template(prompt_path.read_text())

        # Set up LLM interface
        self.llm = ChatOllama(model=model)

    def build_prompt(self, topic: str, conversation: str) -> str:
        return self.template.render(
            agent_name=self.name, topic=topic, conversation=conversation
        )

    def generate_comment(self, topic: str, conversation: str) -> str:
        if self.finished:
            response = f"{self.name}: Nothing to add"
            self._log_transcript(response, conversation)
            return response

        prompt = self.build_prompt(topic, conversation)

        try:
            response = self.llm.invoke(prompt).content.strip()
            print(f"\n🧪 {self.name} raw output:\n{response}\n")
        except Exception as e:
            response = f"{self.name}: Nothing to add  # LLM error: {str(e)}"
            self.finished = True
            self._log_transcript(response, conversation)
            return response

        # Logic for early stopping
        if "Nothing to add" in response or len(response.split()) < 4:
            self.turn_count += 1
            if self.turn_count >= self.min_turns:
                self.finished = True
        else:
            self.turn_count += 1

        self.memory.append(response)
        self._log_transcript(response, conversation)
        return response

    def _log_transcript(self, response: str, conversation: str) -> None:
        """Helper method to log the conversation to transcript files"""
        try:
            # Get current round number
            round_number = self._get_current_round()
            logs_dir = Path(__file__).parent.parent.parent / "logs"

            # Create round-specific directory
            round_dir = logs_dir / f"round_{round_number}"
            round_dir.mkdir(parents=True, exist_ok=True)

            # Write to round-specific transcript
            round_transcript = round_dir / "transcript.txt"
            with open(round_transcript, "a") as f:
                if conversation:
                    f.write(conversation + "\n")
                f.write(response + "\n\n")

        except Exception as e:
            print(f"⚠️ Error writing transcript: {str(e)}")

    def _get_current_round(self) -> int:
        """Helper method to determine current round number"""
        try:
            logs_dir = Path(__file__).parent.parent.parent / "logs"
            existing_rounds = [
                d.name
                for d in logs_dir.iterdir()
                if d.is_dir() and d.name.startswith("round_")
            ]
            return len(existing_rounds) + 1
        except Exception as e:
            print(f"⚠️ Error determining round number: {str(e)}")
            return 1  # Fallback to round 1 if error occurs
