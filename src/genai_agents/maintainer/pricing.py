"""
Pricing Analysis Module

Handles tracking and analysis of API pricing changes.
"""

import logging
from typing import Dict, Any, Optional
import json
import requests
from bs4 import BeautifulSoup
import re
from ..config import Config

logger = logging.getLogger(__name__)


class PricingAnalyzer:
    """Handles API pricing analysis and tracking"""

    def __init__(self, test_mode: bool = False):
        """Initialize the pricing analyzer.

        Args:
            test_mode: Whether to use mock responses
        """
        self.test_mode = test_mode
        self.pricing_data: Dict[str, Dict[str, Any]] = {}
        self._load_pricing_data()

    def _load_pricing_data(self) -> None:
        """Load previous pricing data from disk"""
        try:
            with open(Config.PRICING_DATA_PATH, "r") as f:
                self.pricing_data = json.load(f)
        except FileNotFoundError:
            logger.info("No previous pricing data found")
            self.pricing_data = {}

    def _save_pricing_data(self) -> None:
        """Save current pricing data to disk"""
        try:
            with open(Config.PRICING_DATA_PATH, "w") as f:
                json.dump(self.pricing_data, f)
        except Exception as e:
            logger.error(f"Failed to save pricing data: {str(e)}")
            raise

    def fetch_pricing_data(self, api_name: str) -> Dict[str, Dict[str, float]]:
        """Fetch current pricing data for an API.

        Args:
            api_name: Name of the API to fetch pricing for

        Returns:
            Dictionary mapping model names to their pricing details

        Raises:
            ValueError: If API name is not recognized
            requests.RequestException: If pricing fetch fails
        """
        if self.test_mode:
            return {
                "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
                "gemini-1.0-pro": {"prompt": 0.00025, "completion": 0.0005},
            }

        if api_name not in Config.PRICING_URLS:
            raise ValueError(f"Unknown API: {api_name}")

        try:
            headers = {"User-Agent": Config.USER_AGENT}
            response = requests.get(Config.PRICING_URLS[api_name], headers=headers, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()

            if api_name == "OpenAI":
                return self._extract_openai_pricing(response.text)
            elif api_name == "Google":
                return self._extract_google_pricing(response.text)
            else:
                raise ValueError(f"No pricing extractor for {api_name}")

        except requests.RequestException as e:
            logger.error(f"Failed to fetch pricing data for {api_name}: {str(e)}")
            raise

    def _extract_openai_pricing(self, html_content: str) -> Dict[str, Dict[str, float]]:
        """Extract OpenAI pricing from HTML content.

        Args:
            html_content: HTML content from pricing page

        Returns:
            Dictionary mapping model names to pricing details

        Raises:
            ValueError: If pricing table cannot be found or parsed
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            pricing_data = {}

            # Find the pricing table
            table = soup.find("table")
            if not table:
                raise ValueError("Could not find pricing table")

            # Extract headers
            headers = [th.text.strip() for th in table.find_all("th")]

            # Find column indices
            model_idx = next((i for i, h in enumerate(headers) if "Model" in h), None)
            prompt_idx = next((i for i, h in enumerate(headers) if "Prompt" in h), None)
            completion_idx = next((i for i, h in enumerate(headers) if "Completion" in h), None)

            if None in (model_idx, prompt_idx, completion_idx):
                raise ValueError("Could not find required columns in pricing table")

            # Parse each row
            for row in table.find_all("tr")[1:]:  # Skip header row
                cells = row.find_all("td")
                if len(cells) > max(model_idx, prompt_idx, completion_idx):
                    try:
                        model_name = cells[model_idx].text.strip()
                        prompt_price = float(re.search(r"[-+]?\d*\.\d+|\d+", cells[prompt_idx].text.strip()).group(0))
                        completion_price = float(re.search(r"[-+]?\d*\.\d+|\d+", cells[completion_idx].text.strip()).group(0))

                        pricing_data[model_name] = {"prompt": prompt_price, "completion": completion_price}
                    except (AttributeError, ValueError) as e:
                        logger.warning(f"Could not parse pricing for model {model_name}: {e}")
                        continue

            if not pricing_data:
                raise ValueError("No pricing data could be extracted")

            return pricing_data

        except Exception as e:
            logger.error(f"Failed to extract OpenAI pricing: {str(e)}")
            raise

    def _extract_google_pricing(self, html_content: str) -> Dict[str, Dict[str, float]]:
        """Extract Google AI pricing from HTML content.

        Args:
            html_content: HTML content from pricing page

        Returns:
            Dictionary mapping model names to pricing details

        Raises:
            ValueError: If pricing information cannot be found or parsed
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            pricing_data = {}

            # Find pricing sections (implementation depends on actual page structure)
            pricing_sections = soup.find_all("div", class_="pricing-section")  # Adjust selector as needed

            for section in pricing_sections:
                try:
                    model_name = section.find("h3").text.strip()
                    prices = section.find_all("span", class_="price")  # Adjust selector as needed

                    if len(prices) >= 2:
                        prompt_price = float(re.search(r"[-+]?\d*\.\d+|\d+", prices[0].text.strip()).group(0))
                        completion_price = float(re.search(r"[-+]?\d*\.\d+|\d+", prices[1].text.strip()).group(0))

                        pricing_data[model_name] = {"prompt": prompt_price, "completion": completion_price}
                except (AttributeError, ValueError, IndexError) as e:
                    logger.warning(f"Could not parse pricing section: {e}")
                    continue

            if not pricing_data:
                raise ValueError("No pricing data could be extracted")

            return pricing_data

        except Exception as e:
            logger.error(f"Failed to extract Google pricing: {str(e)}")
            raise

    def get_pricing_changes(self, api_name: str) -> Optional[Dict[str, Any]]:
        """Get changes between current and previous pricing.

        Args:
            api_name: Name of the API to check

        Returns:
            Dictionary of changes or None if no previous data

        Raises:
            ValueError: If API name is not recognized
        """
        if api_name not in self.pricing_data:
            return None

        try:
            current_pricing = self.fetch_pricing_data(api_name)
            previous_pricing = self.pricing_data.get(api_name, {})

            changes = {}

            # Compare prices for each model
            all_models = set(current_pricing.keys()) | set(previous_pricing.keys())

            for model in all_models:
                if model not in previous_pricing:
                    changes[model] = {"status": "new", "pricing": current_pricing[model]}
                elif model not in current_pricing:
                    changes[model] = {"status": "removed", "pricing": previous_pricing[model]}
                else:
                    current = current_pricing[model]
                    previous = previous_pricing[model]

                    if current != previous:
                        changes[model] = {
                            "status": "changed",
                            "old": previous,
                            "new": current,
                            "percent_change": {k: ((current[k] - previous[k]) / previous[k] * 100) for k in current.keys()},
                        }

            return changes if changes else None

        except Exception as e:
            logger.error(f"Failed to get pricing changes for {api_name}: {str(e)}")
            raise

    def update_pricing_data(self, api_name: str) -> Optional[Dict[str, Any]]:
        """Update stored pricing data and return changes.

        Args:
            api_name: Name of the API to update

        Returns:
            Dictionary of changes or None if no changes

        Raises:
            ValueError: If API name is not recognized
        """
        try:
            new_pricing = self.fetch_pricing_data(api_name)
            changes = self.get_pricing_changes(api_name)

            if changes:
                self.pricing_data[api_name] = new_pricing
                self._save_pricing_data()
                logger.info(f"Updated pricing data for {api_name}")

            return changes

        except Exception as e:
            logger.error(f"Failed to update pricing data for {api_name}: {str(e)}")
            raise
