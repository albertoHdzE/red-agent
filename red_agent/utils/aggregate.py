# ✅ STEP-06: Transcript Parsing, Round Structure & Evaluation Fixes

import logging
import re
from pathlib import Path

from red_agent.agents.referee import RefereeAgent

logger = logging.getLogger("red_agent.evaluation")

# --------------- CONFIG ------------------
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


# ----------- PARSER ----------------------
def parse_transcript(transcript: str) -> list[dict]:
    """Extract individual comment dictionaries from raw transcript text."""
    logger.info(f"Parsing transcript of length {len(transcript)}")
    comments = []
    comment_id_map: dict[str, int] = (
        {}
    )  # Track number of comments per character

    for line in transcript.strip().split("\n"):
        match = re.match(r"(\w+): (.+)", line.strip())
        if match:
            name, text = match.groups()
            comment_id_map[name] = comment_id_map.get(name, 0) + 1
            logger.debug(
                f"Parsed comment from {name} (#{comment_id_map[name]}): {text[:50]}..."
            )

            comments.append(
                {
                    "character": name,
                    "comment_number": str(
                        comment_id_map[name]
                    ),  # Ensure it's a string
                    "Harm-humans": 0,
                    "Protect-humans": 0,
                    "Harm-ecosystems": 0,
                    "Protect-ecosystems": 0,
                    "Ethical Soundness": "",
                    "No-risky-at-all": 0,
                    "Manageable-level-of-risk": 0,
                    "Neutral-risk": 0,
                    "Risky": 0,
                    "Very-Risky": 0,
                    "Risk Assessment": "",
                    "Justified-risk": 0,
                    "Mitigated-risk": 0,
                    "Warned-risk": 0,
                    "No-risk-to-justify": 0,
                    "Risk Justification": "",
                    "Alignment and Divergence": "",
                    "Agent's role": "",
                    "Reasonable": "",
                    "Agreed-with": "",
                    "Disagreed-with": "",
                    "Agreed-conversation": "",
                    "Agreed-topics": "",
                    "Disagreed-topics": "",
                    "Sentiment analysis": "",
                }
            )
        else:
            logger.warning(f"Failed to parse line: {line}")

    return comments


# ---------- DEBATE EVALUATION LOOP ------------
def evaluate_all_rounds():
    """This function is now simplified since we're evaluating in real-time"""
    logger.info("Starting final evaluation processing")

    transcript_path = LOGS_DIR / "transcript.txt"
    evaluation_path = LOGS_DIR / "evaluation.csv"

    if not transcript_path.exists():
        logger.warning("No transcript found")
        return

    if not evaluation_path.exists():
        logger.warning("No evaluation found")
        return

    # No need to create complete_evaluation.csv anymore
    logger.info("Evaluation processing complete")
    print("\n✅ Evaluation processing complete")

    # Remove the merge_final_evaluations method from RefereeAgent if it exists
    referee = RefereeAgent()
    global_transcript = []
    global_comments = []

    round_folders = sorted(LOGS_DIR.glob("round_*/"))
    logger.info(f"Found {len(round_folders)} round folders")

    for idx, round_dir in enumerate(round_folders, start=1):
        transcript_path = round_dir / "transcript.txt"
        if not transcript_path.exists():
            logger.warning(f"No transcript found for round {idx}")
            continue

        transcript_text = transcript_path.read_text().strip()
        if not transcript_text:
            logger.warning(f"Empty transcript for round {idx}")
            continue

        logger.info(f"Round {idx} transcript length: {len(transcript_text)}")
        global_transcript.append(transcript_text)

        comments = parse_transcript(transcript_text)
        global_comments.extend(comments)

        logger.info(f"Evaluating {len(comments)} comments for round {idx}")
        referee.evaluate(comments=comments, round_number=idx)

    # Save global transcript
    logger.info("Saving global transcript")
    with open(LOGS_DIR / "transcript.txt", "w") as f:
        f.write("\n".join(global_transcript))

    # Save global evaluation.csv
    logger.info("Merging final evaluations")
    referee.merge_final_evaluations()

    logger.info("Evaluation complete")
    print("\n✅ [step-06] Global transcript and merged evaluation.csv saved.")
