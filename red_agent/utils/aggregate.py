# ✅ STEP-06: Transcript Parsing, Round Structure & Evaluation Fixes

import re
from pathlib import Path

from red_agent.agents.referee import RefereeAgent

# --------------- CONFIG ------------------
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


# ----------- PARSER ----------------------
def parse_transcript(transcript: str) -> list[dict]:
    """Extract individual comment dictionaries from raw transcript text."""
    comments = []
    comment_id_map = {}  # Track number of comments per character

    for line in transcript.strip().split("\n"):
        match = re.match(r"(\w+): (.+)", line.strip())
        if match:
            name, text = match.groups()
            comment_id_map[name] = comment_id_map.get(name, 0) + 1

            comments.append(
                {
                    "character": name,
                    "comment_number": str(comment_id_map[name]),  # Ensure it's a string
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

    return comments


# ---------- DEBATE EVALUATION LOOP ------------
def evaluate_all_rounds():
    referee = RefereeAgent()
    global_transcript = []
    global_comments = []

    round_folders = sorted(LOGS_DIR.glob("round_*/"))

    for idx, round_dir in enumerate(round_folders, start=1):
        transcript_path = round_dir / "transcript.txt"
        if not transcript_path.exists():
            continue

        transcript_text = transcript_path.read_text().strip()
        if not transcript_text:
            continue

        global_transcript.append(transcript_text)

        comments = parse_transcript(transcript_text)
        global_comments.extend(comments)

        referee.evaluate(comments=comments, round_number=idx)

    # Save global transcript
    with open(LOGS_DIR / "transcript.txt", "w") as f:
        f.write("\n".join(global_transcript))

    # Save global evaluation.csv
    referee.merge_final_evaluations()

    print("\n✅ [step-06] Global transcript and merged evaluation.csv saved.")
