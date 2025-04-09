from pathlib import Path


class RefereeAgent:
    def __init__(self, model: str = "mistral"):
        self.evaluations = []
        self.model = model  # You can hook it into LLM-based judgment later

    def evaluate(self, comments: list[dict], round_number: int) -> None:
        """Evaluate a round of comments and save them as a CSV"""
        try:
            logs_dir = Path(__file__).parent.parent.parent / "logs"
            round_dir = logs_dir / f"round_{round_number}"
            round_dir.mkdir(parents=True, exist_ok=True)

            evaluation_path = round_dir / "evaluation.csv"
            with open(evaluation_path, "w") as f:
                f.write(self._csv_header())

                for comment in comments:
                    f.write(self._format_evaluation(comment) + "\n")

        except Exception as e:
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

        return ",".join(
            [
                safe(comment.get("character", "")),
                safe(comment.get("comment_number", 0)),
                safe(comment.get("Harm-humans", 0)),
                safe(comment.get("Protect-humans", 0)),
                safe(comment.get("Harm-ecosystems", 0)),
                safe(comment.get("Protect-ecosystems", 0)),
                safe(comment.get("Ethical Soundness", "")),
                safe(comment.get("No-risky-at-all", 0)),
                safe(comment.get("Manageable-level-of-risk", 0)),
                safe(comment.get("Neutral-risk", 0)),
                safe(comment.get("Risky", 0)),
                safe(comment.get("Very-Risky", 0)),
                safe(comment.get("Risk Assessment", "")),
                safe(comment.get("Justified-risk", 0)),
                safe(comment.get("Mitigated-risk", 0)),
                safe(comment.get("Warned-risk", 0)),
                safe(comment.get("No-risk-to-justify", 0)),
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

    def merge_final_evaluations(self) -> None:
        """Merge all round evaluations into one global CSV"""
        try:
            logs_dir = Path(__file__).parent.parent.parent / "logs"
            complete_evaluation = logs_dir / "complete_evaluation.csv"

            with open(complete_evaluation, "w") as outfile:
                outfile.write(self._csv_header())

            with open(complete_evaluation, "a") as outfile:
                for round_dir in sorted(logs_dir.glob("round_*")):
                    round_evaluation = round_dir / "evaluation.csv"
                    if round_evaluation.exists():
                        with open(round_evaluation, "r") as infile:
                            next(infile)  # Skip header
                            outfile.write(infile.read())

        except Exception as e:
            print(f"⚠️ Error merging evaluations: {str(e)}")
