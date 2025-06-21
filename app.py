"""
Blog Post Generator Launcher

This script initializes logging, loads environment variables, 
and launches the Gradio UI for generating blog posts using the 
blog_generator utility functions.

Modules:
- dotenv: Loads environment variables from a .env file.
- logging: Enables logging for debugging and monitoring.
- blog_generator.utils: Contains the `create_ui` function that builds the Gradio interface.
"""

from dotenv import load_dotenv
import logging
from blog_generator.utils import create_ui

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blog_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BlogPostGenerator")

# Load environment variables
load_dotenv()


def main():
    """
    Main function to run the blog post generator UI.

    - Initializes and launches the Gradio web interface.
    - UI is created via the `create_ui` function from `blog_generator.utils`.
    """
    demo = create_ui()
    demo.launch()


if __name__ == "__main__":
    main()
