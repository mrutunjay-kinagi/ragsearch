"""
config.py - Load and parse configuration data from a JSON
"""
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_configuration(file_path: str) -> dict:
    """
    Load and parse configuration data from a JSON or YAML file.

    Args:
        file_path (str): The path to the configuration file.
    Returns:
        dict: The parsed configuration data.
    Raises:
        Exception: If an error occurs while loading the configuration.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        logging.info(f"Configuration loaded from {file_path}.")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration from {file_path}: {e}")
        raise
