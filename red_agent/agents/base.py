class DebateAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = []
        self.finished = False

    def generate_comment(self, topic: str, conversation: str) -> str:
        """
        Generate a response in 100 words max based on topic and full
        conversation.
        """
        prompt = self.build_prompt(topic, conversation)
        # Stub - you’ll plug LLM call here
        return f"{self.name}: Nothing to add"  # placeholder

    def build_prompt(self, topic: str, conversation: str) -> str:
        return f"""
You are {self.name}, participating in a critical debate. The topic is:

"{topic}"

Respond in this format:
{self.name}: your argument here.

You can agree/disagree with others, and stop with "Nothing to add".
You must not exceed 100 words.

Conversation so far:
{conversation}
        """.strip()
