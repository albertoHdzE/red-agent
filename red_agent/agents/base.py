import logging
from pathlib import Path

from jinja2 import Template
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger("red_agent.agents")


class DebateAgent:
    memory: list[str]

    def __init__(
        self,
        name: str,
        role: str,
        model: str,
        description: str,
        min_turns: int,
        max_turns: int,
        active_agent_names: list[str] | None = None,
    ):
        self.name: str = name
        self.role: str = role
        self.description: str = description
        self.memory = []
        self.finished: bool = False
        self.model_name: str = model
        self.turn_count: int = 0
        self.min_turns: int = min_turns
        self.max_turns: int = max_turns
        self.active_agent_names = active_agent_names or []
        self._llm = None  # Don't initialize the LLM here

        logger.info(
            f"Initializing agent {name} with role {role} using model {model}, active agents: {self.active_agent_names}"
        )

        prompt_path = (
            Path(__file__).parent.parent / "prompts/debate_prompt.jinja"
        )
        self.template: Template = Template(prompt_path.read_text())

        logger.info(
            f"Agent {name} initialized successfully (model will be loaded on demand)"
        )

    @property
    def llm(self):
        """Lazy-load the LLM model only when needed"""
        if self._llm is None:
            logger.info(
                f"Loading model {self.model_name} for agent {self.name}"
            )
            self._llm = ChatOllama(model=self.model_name)
        return self._llm

    def release_model(self):
        """Release the model to free memory"""
        if self._llm is not None:
            logger.info(
                f"Releasing model {self.model_name} for agent {self.name}"
            )
            self._llm = None
            # Force garbage collection to ensure memory is freed
            import gc

            gc.collect()

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

        full_conversation = conversation
        logger.debug(
            f"Conversation length for {self.name}: {len(full_conversation)} characters"
        )

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
            for other_agent_name in [
                name for name in self.active_agent_names if name != self.name
            ]:
                other_agent_prefix = f"{other_agent_name}:"
                if other_agent_prefix in response:
                    logger.warning(
                        f"Response from {self.name} contains another agent's prefix: {other_agent_prefix}, removing it"
                    )
                    response = response.replace(other_agent_prefix, "")

            # Enforce word count limit (approximately 100 words)
            words = response.split()
            if len(words) > 105:
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
            # Make sure to release the model even if there's an error
            self.release_model()
            return response

        # Logic for early stopping
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
        try:
            logger.debug(f"Logging to transcript: {response[:50]}...")
        except Exception as e:
            logger.error(f"Error logging transcript: {str(e)}")
