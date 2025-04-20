import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.chat_models import ChatOllama

logger = logging.getLogger("red_agent.agents.referee")


class RefereeAgent:
    def __init__(self, model: str = "deepseek-coder-v2:latest"):
        self.model_name = model
        self.llm = ChatOllama(model=self.model_name)
        prompt_path = (
            Path(__file__).parent.parent / "prompts/referee_prompt.txt"
        )
        self.prompt_template = prompt_path.read_text()
        # Get output template keys for CSV header
        output_template_path = (
            Path(__file__).parent.parent
            / "prompts/evaluation_output_template.jinja"
        )
        self.output_keys = self._parse_output_template_keys(
            output_template_path.read_text()
        )

    def _parse_output_template_keys(self, template_text: str) -> List[str]:
        # Extract keys from the Jinja template (lines like: key: {{ value }})
        keys = []
        for line in template_text.splitlines():
            if ":" in line:
                key = line.split(":")[0].strip()
                if key:
                    keys.append(key)
        return keys

    def _parse_comment_blocks(self, transcript_text: str) -> List[str]:
        # Split transcript into comments, each starting with agent name and colon
        pattern = re.compile(r"^[A-Za-z0-9_]+:.*", re.MULTILINE)
        matches = list(pattern.finditer(transcript_text))
        comments = []
        for i, match in enumerate(matches):
            start = match.start()
            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(transcript_text)
            )
            comment = transcript_text[start:end].strip()
            if comment:
                comments.append(comment)
        return comments

    def _format_prompt(self, comment: str) -> str:
        # Insert the comment into the referee prompt template
        return self.prompt_template.replace(
            "[here is the comment(s) to evaluate]", comment
        )

    def _parse_llm_output(self, output: str) -> Dict[str, Any]:
        # Parse the output according to the evaluation_output_template.jinja
        # Each line is "key: value"
        result = {}
        for line in output.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip().strip('"')
        return result

    def evaluate_transcript(
        self, transcript_path: Path, evaluation_csv_path: Path
    ):
        logger.info(
            f"RefereeAgent: Evaluating transcript at {transcript_path}"
        )
        transcript_text = transcript_path.read_text()
        comments = self._parse_comment_blocks(transcript_text)
        logger.info(f"Found {len(comments)} comments to evaluate.")

        # Prepare CSV
        file_exists = evaluation_csv_path.exists()
        with evaluation_csv_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.output_keys)
            if not file_exists:
                writer.writeheader()

            for idx, comment in enumerate(comments, 1):
                logger.info(f"Evaluating comment {idx}/{len(comments)}")
                while True:
                    try:
                        prompt = self._format_prompt(comment)
                        llm_response = self.llm.invoke(prompt)
                        output = str(llm_response.content).strip()
                        parsed = self._parse_llm_output(output)
                        # Add comment_number if not present
                        if (
                            "comment_number" in self.output_keys
                            and "comment_number" not in parsed
                        ):
                            parsed["comment_number"] = str(idx)
                        # Ensure all keys are present
                        row = {
                            key: parsed.get(key, "")
                            for key in self.output_keys
                        }
                        writer.writerow(row)
                        logger.info(
                            f"Evaluation for comment {idx} written to CSV."
                        )
                        break
                    except Exception as e:
                        logger.error(
                            f"Error evaluating comment {idx}: {e}. Retrying..."
                        )
