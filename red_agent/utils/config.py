import logging
from pathlib import Path

import yaml  # type: ignore

logger = logging.getLogger("red_agent.config")


def load_config():
    """Load agent and debate configuration from YAML file"""
    try:
        # Update path to point to project root
        config_path = (
            Path(__file__).parent.parent.parent / "agents_config.yaml"
        )
        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(
            f"Loaded configuration with {len(config['agents'])} agents"
        )
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
        raise
