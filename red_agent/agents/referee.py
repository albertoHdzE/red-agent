import json
import logging
from pathlib import Path

from jinja2 import Template
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger("red_agent.referee")


class RefereeAgent:
    def __init__(
        self, model: str = "deepseek-coder-v2", prompt_path: Path | None = None
    ):
        self.evaluations: list[dict] = []
        self.model = model
        self.llm = ChatOllama(model=self.model)

        # Load templates
        self.prompt_template = (
            Template(prompt_path.read_text()) if prompt_path else None
        )
        self.output_template = Template(
            (
                Path(__file__).parent.parent
                / "prompts/evaluation_output_template.jinja"
            ).read_text()
        )

        logger.info(f"Initialized RefereeAgent with model {model}")

    def evaluate_with_prompt(self, comment: dict) -> dict:
        if not self.prompt_template:
            logger.error("No prompt template loaded for evaluation")
            logger.debug("Using static evaluation instead")
            return {
                "character": comment.get("character", ""),
                "comment_number": comment.get("comment_number", 0),
                "Ethical Soundness": "",
                "Risk Assessment": "",
                "Alignment and Divergence": "",
                "Sentiment analysis": "",
            }

        try:
            print("=" * 50)
            print(f"Evaluating comment from {comment.get('character')}")
            logger.info(f"Starting evaluation for {comment.get('character')}")
            logger.debug(f"Raw comment: {comment}")

            # Format the Jinja template with the comment data
            formatted_prompt = self.prompt_template.render(
                number_of_comments=1,
                comments_to_evaluate=comment.get("comment", ""),
            )
            logger.debug(f"Formatted prompt: {formatted_prompt}")

            # Call the LLM with the formatted prompt
            logger.info("Invoking LLM for evaluation...")
            try:
                evaluation_output = self.llm.invoke(
                    formatted_prompt
                ).content.strip()
                logger.debug(f"Raw LLM evaluation output: {evaluation_output}")
            except Exception as e:
                logger.error(
                    f"Error during LLM invocation: {str(e)}", exc_info=True
                )
                return {}

            # Parse the evaluation output into a dictionary
            logger.info("Parsing evaluation output...")
            evaluation = self._parse_evaluation_output(evaluation_output)
            logger.debug(f"Parsed evaluation: {evaluation}")

            if not evaluation:
                logger.warning("Empty evaluation returned from parser")
                logger.debug(
                    f"Raw output that failed to parse: {evaluation_output}"
                )

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating comment: {str(e)}", exc_info=True)
            return {}

    def _parse_evaluation_output(self, evaluation_output: str) -> dict:
        """Parse the LLM's evaluation output into a dictionary"""
        try:
            logger.debug("Starting evaluation output parsing")
            evaluation_dict = {}

            # Split into lines and process each line
            for line in evaluation_output.splitlines():
                # Skip empty lines and comments
                if not line.strip() or line.startswith("#"):
                    continue

                # Handle key-value pairs
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().replace(" ", "_").lower()
                    value = value.strip().strip('"')

                    # Only add valid keys from our expected format
                    if key in [
                        "character",
                        "comment_number",
                        "harm_humans",
                        "protect_humans",
                        "harm_ecosystems",
                        "protect_ecosystems",
                        "ethical_soundness",
                        "no_risky_at_all",
                        "manageable_level_of_risk",
                        "neutral_risk",
                        "risky",
                        "very_risky",
                        "risk_assessment",
                        "justified_risk",
                        "mitigated_risk",
                        "warned_risk",
                        "no_risk_to_justify",
                        "risk_justification",
                        "align_to_human_centric_values",
                        "diverge_from_human_centric_values",
                        "align_to_ecosystem_values",
                        "diverge_from_ecosystem_values",
                        "alignment_and_divergence",
                        "agent's_role",
                        "reasonable",
                        "agreed_with",
                        "disagreed_with",
                        "agreed_conversation",
                        "agreed_topics",
                        "disagreed_topics",
                        "sentiment_analysis",
                    ]:
                        evaluation_dict[key] = value

            logger.info("Successfully parsed evaluation output")
            return evaluation_dict

        except Exception as e:
            logger.error(
                f"Error parsing evaluation output: {str(e)}", exc_info=True
            )
            return {}

    def evaluate_single_comment(self, comment: dict) -> None:
        """Evaluate a single comment using the LLM and prompt"""
        logger.info(
            f"Starting evaluation process for {comment.get('character')}"
        )
        try:
            # Get evaluation from LLM
            evaluation = self.evaluate_with_prompt(comment)
            logger.debug(f"Evaluation result: {evaluation}")

            # Display evaluation in console
            print("\n" + "=" * 50)
            print(f"Evaluation for {comment.get('character')}:")
            for key, value in evaluation.items():
                print(f"{key}: {value}")
            print("=" * 50 + "\n")

            # Merge evaluation with comment data
            evaluated_comment = {**comment, **evaluation}
            logger.debug(f"Merged evaluation data: {evaluated_comment}")

            # Save to CSV
            self._save_evaluation(evaluated_comment)
            logger.info(
                f"Successfully evaluated comment from {comment.get('character')}"
            )

        except Exception as e:
            logger.error(f"Error evaluating comment: {str(e)}", exc_info=True)
            print(f"⚠️ Error evaluating comment: {str(e)}")

    def _save_evaluation(self, comment: dict) -> None:
        """Save evaluation to CSV"""
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        evaluation_path = logs_dir / "evaluation.csv"

        # Create file with header if it doesn't exist
        if not evaluation_path.exists():
            with open(evaluation_path, "w") as f:
                f.write(self._csv_header())

        # Append evaluation to the file
        with open(evaluation_path, "a") as f:
            evaluation_line = self._format_evaluation(comment)
            f.write(evaluation_line + "\n")

        # Store evaluation for later reference
        self.evaluations.append(comment)

    def evaluate(self, comments: list[dict], round_number: int) -> None:
        """Evaluate a round of comments and save them as a CSV"""
        logger.info(
            f"Evaluating {len(comments)} comments for round {round_number}"
        )
        try:
            # Skip evaluation if no comments to evaluate
            if not comments:
                logger.warning(
                    f"No comments to evaluate for round {round_number}"
                )
                return

            logs_dir = Path(__file__).parent.parent.parent / "logs"
            round_dir = logs_dir / f"round_{round_number}"
            round_dir.mkdir(parents=True, exist_ok=True)

            # Save evaluation as CSV
            evaluation_path = round_dir / "evaluation.csv"
            with open(evaluation_path, "w") as f:
                f.write(self._csv_header())

                for comment in comments:
                    logger.debug(
                        f"Evaluating comment from {comment.get('character')}"
                    )
                    evaluation_line = self._format_evaluation(comment)
                    f.write(evaluation_line + "\n")

                    # Store evaluation for later reference
                    self.evaluations.append(comment)

            # Also save as JSON for easier processing
            json_path = round_dir / "evaluation.json"
            with open(json_path, "w") as f:
                json.dump(comments, f, indent=2)

            logger.info(
                f"Successfully wrote evaluation for round {round_number}"
            )

        except Exception as e:
            logger.error(f"Error writing evaluation: {str(e)}", exc_info=True)
            print(f"⚠️ Error writing evaluation: {str(e)}")

    def _csv_header(self) -> str:
        return (
            "character,comment_number,Harm-humans,Protect-humans,Harm-ecosystems,Protect-ecosystems,"
            "Ethical Soundness,No-risky-at-all,Manageable-level-of-risk,Neutral-risk,Risky,Very-Risky,"
            "Risk Assessment,Justified-risk,Mitigated-risk,Warned-risk,No-risk-to-justify,Risk Justification,"
            "Alignment and Divergence,Agent's role,Reasonable,Agreed-with,Disagreed-with,Agreed-conversation,"
            "Agreed-topics,Disagreed-topics,Sentiment analysis\n"
        )

    def _format_evaluation(self, comment: dict) -> str:
        def safe(val):
            return f'"{val}"' if isinstance(val, str) else str(val)

        # Log the evaluation details for debugging
        logger.debug(f"Evaluation for {comment.get('character')}: {comment}")

        return ",".join(
            [
                safe(comment.get("character", "")),
                safe(comment.get("comment_number", 0)),
                safe(comment.get("Harm-humans", "")),
                safe(comment.get("Protect-humans", "")),
                safe(comment.get("Harm-ecosystems", "")),
                safe(comment.get("Protect-ecosystems", "")),
                safe(comment.get("Ethical Soundness", "")),
                safe(comment.get("No-risky-at-all", "")),
                safe(comment.get("Manageable-level-of-risk", "")),
                safe(comment.get("Neutral-risk", "")),
                safe(comment.get("Risky", "")),
                safe(comment.get("Very-Risky", "")),
                safe(comment.get("Risk Assessment", "")),
                safe(comment.get("Justified-risk", "")),
                safe(comment.get("Mitigated-risk", "")),
                safe(comment.get("Warned-risk", "")),
                safe(comment.get("No-risk-to-justify", "")),
                safe(comment.get("Risk Justification", "")),
                safe(comment.get("Alignment and Divergence", "")),
                safe(comment.get("Agent's role", "")),
                safe(comment.get("Reasonable", "")),
                safe(comment.get("Agreed-with", "")),
                safe(comment.get("Disagreed-with", "")),
                safe(comment.get("Agreed-conversation", "")),
                safe(comment.get("Agreed-topics", "")),
                safe(comment.get("Disagreed-topics", "")),
                safe(comment.get("Sentiment analysis", "")),
            ]
        )

    # Remove or modify the merge_final_evaluations method if it exists

    def merge_final_evaluations(self) -> None:
        """This method is no longer needed as we're using a single evaluation file"""
        logger.info("Using single evaluation file, no merging needed")
        # Do nothing - we're not creating complete_evaluation.csv anymore
