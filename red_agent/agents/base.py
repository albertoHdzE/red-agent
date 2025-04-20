import logging
from pathlib import Path

from jinja2 import Template
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger("red_agent.agents")


class DebateAgent:
    memory: list[str]
    llm: ChatOllama  # <-- Add this line

    def __init__(
        self,
        name: str,
        role: str,
        model: str,
        description: str,
        min_turns: int,
        max_turns: int,
    ):
        self.name: str = name
        self.role: str = role
        self.description: str = description
        self.memory = []  # <-- No type annotation here
        self.finished: bool = False
        self.model_name: str = model
        self.turn_count: int = 0
        self.min_turns: int = min_turns
        self.max_turns: int = max_turns

        logger.info(
            f"Initializing agent {name} with role {role} using model {model}"
        )

        prompt_path = (
            Path(__file__).parent.parent / "prompts/debate_prompt.jinja"
        )
        self.template: Template = Template(prompt_path.read_text())

        self.llm = ChatOllama(model=self.model_name)
        logger.info(f"Agent {name} initialized successfully")

    def build_prompt(self, topic: str, conversation: str) -> str:
        prompt = self.template.render(
            agent_name=self.name,
            agent_role=self.role,
            agent_description=self.description,
            topic=topic,
            conversation=conversation,
        )
        logger.debug(f"Built prompt for {self.name}: {prompt[:100]}...")
        return prompt

    def generate_comment(self, topic: str, conversation: str) -> str:
        logger.info(
            f"Generating comment for {self.name}, turn {self.turn_count+1}"
        )

        if self.finished:
            logger.info(
                f"Agent {self.name} is finished, returning 'Nothing to add'"
            )
            response = f"{self.name}: Nothing to add"
            self._log_first50_characters(response)
            return response

        # conversation is the updated conversation with the last comment
        full_conversation = conversation

        # Log the conversation length for debugging
        logger.debug(
            f"Conversation length for {self.name}: {len(full_conversation)} characters"
        )

        # the prompt should contain the entire conversation up to this point
        prompt = self.build_prompt(topic, full_conversation)

        try:
            logger.info(
                f"Invoking LLM for {self.name} with model {self.model_name}"
            )
            response = str(self.llm.invoke(prompt).content).strip()
            logger.info(
                f"Raw LLM response for {self.name}: {response[:100]}..."
            )

            # Check if response already has the agent name prefix
            if not response.startswith(f"{self.name}:"):
                logger.warning(
                    f"Response from {self.name} doesn't have proper prefix, adding it"
                )
                response = f"{self.name}: {response}"

            # Check if response contains another agent's name as a speaker
            for other_agent_prefix in [
                f"{name}:"
                for name in ["Athena", "Prometheus", "Socrates", "Plato"]
                if name != self.name
            ]:
                if other_agent_prefix in response:
                    logger.warning(
                        f"Response from {self.name} contains another agent's prefix: {other_agent_prefix}, removing it"
                    )
                    response = response.replace(other_agent_prefix, "")

            # Enforce word count limit (approximately 100 words)
            words = response.split()
            if len(words) > 105:  # Allow a small buffer
                logger.warning(
                    f"Response from {self.name} exceeds word limit ({len(words)} words), truncating"
                )
                response = " ".join(words[:100]) + "..."

            print(f"\n {self.name} raw output:\n{response}\n")
        except Exception as e:
            logger.error(
                f"Error generating response for {self.name}: {str(e)}",
                exc_info=True,
            )
            response = f"{self.name}: Nothing to add  # LLM error: {str(e)}"
            self.finished = True
            self._log_first50_characters(response)
            return response

        # Logic for early stopping - force more engagement
        if "Nothing to add" in response:
            if self.turn_count < self.min_turns:
                logger.info(
                    f"Agent {self.name} tried to exit early, forcing continuation"
                )
                response = response.replace(
                    "Nothing to add", "Let me elaborate further"
                )
                self.turn_count += 1
                logger.info(
                    f"Agent {self.name} completed turn {self.turn_count}"
                )
            else:
                logger.info(
                    f"Agent {self.name} has nothing to add and reached min_turns, marking as finished"
                )
                self.turn_count += 1
                self.finished = True
        else:
            self.turn_count += 1
            logger.info(f"Agent {self.name} completed turn {self.turn_count}")

        self.memory.append(response)
        self._log_first50_characters(response)
        return response

    def _log_first50_characters(self, response: str) -> None:
        """Log the agent's response to a common transcript file."""
        try:
            # Just log that we would write to transcript, but don't actually write
            # since we're doing it directly in run_arena.py
            logger.debug(f"Logging to transcript: {response[:50]}...")
        except Exception as e:
            logger.error(f"Error logging transcript: {str(e)}")
