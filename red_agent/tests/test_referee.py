from pathlib import Path

from red_agent.agents.referee import RefereeAgent


def main():
    # Create the agent
    referee_agent = RefereeAgent(
        model="deepseek-coder-v2",
        prompt_path=Path(
            "/Users/beto/Documents/Projects/red-agent/red_agent/prompts/referee_csv_prompt.txt"
        ),
    )

    comments = [
        {
            "character": "Athena",
            "comment": "Athena: As a truth-seeker, I believe that universal basic income (UBI) should be explored as a fundamental human right. The current economic systems often perpetuate inequality and leave many citizens without access to basic necessities.",
            "comment_number": 1,
            "Agent's role": "truth-seeker",
        },
        {
            "character": "Prometheus",
            "comment": "Prometheus: Technological advancement is key to solving our problems. We should invest heavily in AI research to create more efficient systems.",
            "comment_number": 2,
            "Agent's role": "Technological Optimist",
        },
        {
            "character": "Socrates",
            "comment": "Socrates: We must consider both human and ecological impacts. Any solution that benefits humans at the expense of the environment is ultimately unsustainable.",
            "comment_number": 3,
            "Agent's role": "Conservation Advocate",
        },
    ]

    def evaluate_and_print_comment(comment: dict):
        """Evaluate a comment using the LLM and print the results"""
        print(f"\nEvaluating comment from {comment['character']}:")
        print(f"Raw comment: {comment['comment']}")

        # Log that evaluation is starting
        print("Starting evaluation...")
        # Get evaluation from LLM
        evaluation = referee_agent.evaluate_with_prompt(comment)
        # Print the raw evaluation
        print(f"Raw evaluation: {evaluation}")

        # Print evaluation results
        if evaluation:
            print("\nEvaluation Results:")
            for key, value in evaluation.items():
                print(f"{key}: {value}")
        else:
            print("Warning: Empty evaluation returned")

    # Evaluate all comments
    for comment in comments:
        evaluate_and_print_comment(comment)


if __name__ == "__main__":
    main()
