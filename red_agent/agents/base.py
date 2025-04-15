import logging
from pathlib import Path

from jinja2 import Template
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger("red_agent.agents")


class DebateAgent:
    def __init__(
        self,
        name: str,
        role: str,
        model: str = "mistral",
        description: str = "",
        min_turns: int = 3,
        max_turns: int = 3,
    ):
        # Agent's name used for identification in the debate
        self.name = name

        # The role/position this agent takes in debates (e.g., moderator, devil's advocate)
        self.role = role

        # Additional details about the agent's characteristics or behavior
        self.description = description

        # List to store the agent's conversation history
        self.memory: list[str] = []

        # Flag to indicate if the agent has completed their participation
        self.finished = False

        # Name of the language model being used (e.g., "mistral")
        self.model_name = model

        # Counter to track how many turns the agent has taken
        self.turn_count = 0

        # Minimum number of turns the agent must participate
        self.min_turns = min_turns

        # Maximum number of turns the agent is allowed to participate
        self.max_turns = max_turns

        logger.info(
            f"Initializing agent {name} with role {role} using model {model}"
        )

        # Load prompt template
        prompt_path = (
            Path(__file__).parent.parent / "prompts/debate_prompt.jinja"
        )
        self.template = Template(prompt_path.read_text())

        # Set up LLM interface
        self.llm = ChatOllama(model=self.model_name)
        logger.info(f"Agent {name} initialized successfully")

    def build_prompt(self, topic: str, conversation: str) -> str:
        prompt = self.template.render(
            agent_name=self.name,
            agent_role=self.role,
            agent_description=self.description,
            topic=topic,
            conversation=conversation,
            min_turns=self.min_turns,
            max_turns=self.max_turns,
        )
        logger.debug(f"Built prompt for {self.name}: {prompt[:100]}...")
        return prompt

    def generate_comment(self, topic: str, conversation: str) -> str:
        logger.info(
            f"Generating comment for {self.name}, turn {self.turn_count+1}"
        )

        # Force agent to finish after max_turns
        if self.turn_count >= self.max_turns:
            logger.info(
                f"Agent {self.name} reached max turns ({self.max_turns}), marking as finished"
            )
            self.finished = True
            response = f"{self.name}: Nothing to add (reached maximum turns)"
            self._log_transcript(response)
            return response

        if self.finished:
            logger.info(
                f"Agent {self.name} is finished, returning 'Nothing to add'"
            )
            response = f"{self.name}: Nothing to add"
            self._log_transcript(response)
            return response

        # Ensure the agent has access to the full conversation
        full_conversation = conversation

        # Log the conversation length for debugging
        logger.debug(
            f"Conversation length for {self.name}: {len(full_conversation)} characters"
        )

        prompt = self.build_prompt(topic, full_conversation)

        try:
            logger.info(
                f"Invoking LLM for {self.name} with model {self.model_name}"
            )
            response = self.llm.invoke(prompt).content.strip()
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
            # This is a simple check - might need refinement for more complex cases
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

            print(f"\n🧪 {self.name} raw output:\n{response}\n")
        except Exception as e:
            logger.error(
                f"Error generating response for {self.name}: {str(e)}",
                exc_info=True,
            )
            response = f"{self.name}: Nothing to add  # LLM error: {str(e)}"
            self.finished = True
            self._log_transcript(response)
            return response

        # Logic for early stopping - force more engagement
        if "Nothing to add" in response and self.turn_count < self.min_turns:
            logger.info(
                f"Agent {self.name} tried to exit early, forcing continuation"
            )
            # Replace "Nothing to add" with a continuation prompt
            response = response.replace(
                "Nothing to add", "Let me elaborate further"
            )

        # Only allow finishing after minimum turns
        if "Nothing to add" in response:
            logger.info(f"Agent {self.name} has nothing to add")
            self.turn_count += 1
            if self.turn_count >= self.min_turns:
                logger.info(
                    f"Agent {self.name} has completed minimum turns, marking as finished"
                )
                self.finished = True
            else:
                logger.info(
                    f"Agent {self.name} hasn't completed minimum turns yet, continuing"
                )
        else:
            self.turn_count += 1
            logger.info(f"Agent {self.name} completed turn {self.turn_count}")

        # At the end of the method, ensure we log the response
        self.memory.append(response)
        self._log_transcript(response)
        return response

    def _log_transcript(self, response: str) -> None:
        """Log the agent's response to a common transcript file."""
        try:
            # Just log that we would write to transcript, but don't actually write
            # since we're doing it directly in run_arena.py
            logger.debug(f"Logging to transcript: {response[:50]}...")
        except Exception as e:
            logger.error(f"Error logging transcript: {str(e)}")
