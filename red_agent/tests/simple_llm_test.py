import logging
import sys
from typing import List

import psutil
from langchain_community.chat_models import ChatOllama

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("simple_llm_test")


def get_memory_usage() -> str:
    """Get current memory usage of the process."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB"


def create_query(
    comments: List[str], num_comments: int, prompt_template: str
) -> str:
    """Create query with specified number of comments."""
    if num_comments > len(comments):
        logger.warning(
            f"Requested {num_comments} comments but only {len(comments)} available"
        )
        num_comments = len(comments)

    selected_comments = comments[:num_comments]
    comments_text = "\n".join(
        f"{i+1}. {comment}" for i, comment in enumerate(selected_comments)
    )

    query = prompt_template.replace(
        "[number of comments]", str(len(selected_comments))
    )
    query = query.replace(
        "[here is the comment(s) to evaluate]", comments_text
    )

    return query


def main():
    # Initialize the LLM with the deepseek-coder-v2 model
    logger.info("Initializing LLM with deepseek-coder-v2 model...")
    logger.debug(f"Initial {get_memory_usage()}")

    llm = ChatOllama(model="deepseek-coder-v2")
    logger.info("LLM initialized successfully.")
    logger.debug(f"After LLM initialization: {get_memory_usage()}")

    comments = [
        "Implementing stricter regulations and enforcing existing laws for industries that consume natural resources and abuse non-human animals, including licensing requirements, fines, and sustainable practice incentives.",
        "Promoting renewable energy sources like solar, wind, and hydro to reduce finite resource demand and pollution, with government investment in R&D and consumer incentives.",
        "Encouraging responsible consumption through awareness of environmental impact, promoting fair trade, boycotting exploitative companies, and supporting ethical businesses.",
        "Protecting biodiversity by establishing protected areas, national parks, and wildlife reserves, while focusing on habitat preservation and ecosystem restoration.",
        "Promoting plant-based diets and reducing animal agriculture through advocacy campaigns and educational programs about veganism and vegetarianism.",
        "Supporting animal welfare through stricter regulations on factory farms and promoting responsible pet ownership education.",
        "Encouraging technological innovation in resource monitoring, pollution detection, and ecosystem management, with investment in green and circular economy technologies.",
        "Fostering international cooperation to address climate change, biodiversity loss, and animal welfare through global agreements and standards.",
        "Addressing population growth through education about contraception, reproductive healthcare access, and family planning while respecting individual rights.",
        "Supporting grassroots activism and community-driven projects to create positive change and advocate for systemic environmental reforms.",
    ]

    # Number of comments to include in the query
    num_comments_to_process = 1  # Can be modified as needed

    # Read the prompt template from file
    prompt_file = "/Users/beto/Documents/Projects/red-agent/red_agent/prompts/referee_csv_prompt.txt"
    try:
        with open(prompt_file, "r") as f:
            prompt_template = f.read()
            logger.debug(
                f"After reading prompt template: {get_memory_usage()}"
            )
    except Exception as e:
        logger.error(f"Error reading prompt file: {str(e)}")
        return

    # Create query with specified number of comments
    logger.info(f"Creating query with {num_comments_to_process} comments...")
    query = create_query(comments, num_comments_to_process, prompt_template)
    logger.debug(f"Query size: {sys.getsizeof(query)} bytes")
    logger.debug(f"After query creation: {get_memory_usage()}")

    logger.info("Sending query to LLM...")
    try:
        response = llm.invoke(query).content.strip()
        logger.info("Received response from LLM.")
        logger.debug(f"After LLM response: {get_memory_usage()}")

        # Print the response
        print("\nLLM Response:")
        print(response)
        print("-" * 80)  # Separator

    except Exception as e:
        logger.error(
            f"Error during LLM invocation: {str(e)}",
            exc_info=True,
        )


if __name__ == "__main__":
    main()
