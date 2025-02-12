## Project Overview

This project is a Python-based AI-powered code maintenance agent that automates tasks such as deprecation checks, security compliance scanning, and model performance monitoring.

## Architecture

The project follows a modular architecture with the following main components:

*   `MaintainerAgent.py`: The core agent that orchestrates the maintenance tasks.
*   `Google.py`: Integration with Google Gemini for text embeddings and similarity analysis.
*   `OpenAI.py`: Integration with OpenAI for language model tasks.

## Coding Guidelines

*   Use descriptive variable and function names.
*   Follow PEP 8 coding style.
*   Write unit tests for all new features.
*   Use type hints for all function arguments and return values.

## Dependencies

*   Python 3.9+
*   PyGithub
*   google-generativeai
*   openai
*   apscheduler
*   pydantic
*   pytest
*   pytest-cov
*   cryptography
*   bleach
*   functools
*   ast
*   inspect