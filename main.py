"""
Main application entry point.
"""

import logging
from config import Config
from OpenAI import OpenAIService
from Google import GoogleAIService
from MaintainerAgent import MaintainerAgent

# Set up logging
logger = Config.setup_logging()


def main():
    """Main application entry point"""
    try:
        # Initialize services
        openai = OpenAIService(test_mode=True)
        google = GoogleAIService(test_mode=True)

        # Use services
        openai.index_document("Some text")
        results = openai.semantic_search("query")
        response = openai.generate_completion("prompt")

        # Run agent
        openai.setup_agent()
        result = openai.run_agent("What can you find about Mars?")

        # Initialize maintainer agent
        maintainer = MaintainerAgent(google_ai_system=google, openai_system=openai, test_mode=True)

        # Add sample documents
        documents = [
            "The Mars rover Perseverance landed in February 2021.",
            "Neural networks are a key component of deep learning.",
            "Python is a popular programming language for AI development.",
        ]

        for doc in documents:
            openai.index_document(doc)
            google.index_document(doc)

        # Test search functionality
        query = "When did the Mars mission land?"
        openai_results = openai.semantic_search(query)
        google_results = google.semantic_search(query)

        logger.info("OpenAI Search Results:")
        for doc, score in openai_results:
            logger.info(f"- {doc} (score: {score:.2f})")

        logger.info("\nGoogle Search Results:")
        for doc, score in google_results:
            logger.info(f"- {doc} (score: {score:.2f})")

        # Test RAG functionality
        question = "What Mars rover are we talking about?"
        openai_answer = openai.generate_completion(question)
        google_answer = google.generate_completion(question)

        logger.info(f"\nOpenAI Answer: {openai_answer}")
        logger.info(f"Google Answer: {google_answer}")

        # Run maintenance cycle
        maintainer.run_maintenance_cycle()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
