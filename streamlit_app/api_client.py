import requests
import logging
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = "https://web-production-87ad8.up.railway.app"
REQUEST_TIMEOUT = 30  # seconds

class APIClient:
    """Client for interacting with the Rainfall Forecast API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = REQUEST_TIMEOUT
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            if response.content:
                return response.json()
            else:
                return {}
                
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {method} {url}")
            raise Exception(f"Request timeout. The API server is taking too long to respond.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for {method} {url}")
            raise Exception(f"Connection error. Could not connect to the API server.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {method} {url}: {e}")
            if response.status_code == 404:
                raise Exception("Endpoint not found. Please check the API URL.")
            elif response.status_code == 500:
                raise Exception("Server error. The API server encountered an internal error.")
            else:
                raise Exception(f"HTTP error {response.status_code}: {response.text}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from {method} {url}")
            raise Exception("Invalid response format from API.")
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_available_states(self) -> List[str]:
        """Get list of available states from the API."""
        try:
            response = self._make_request("GET", "/states")
            
            # Handle different response formats
            if isinstance(response, list):
                return response
            elif isinstance(response, dict) and "states" in response:
                return response["states"]
            elif isinstance(response, dict) and "data" in response:
                return response["data"]
            else:
                logger.warning(f"Unexpected response format: {response}")
                return list(response.keys()) if response else []
                
        except Exception as e:
            logger.error(f"Error getting states: {e}")
            raise Exception(f"Failed to get available states: {str(e)}")
    
    def get_forecast(self, state: str, start_date: str) -> List[Dict[str, Any]]:
        """Get rainfall forecast from the API."""
        try:
            payload = {
                "state_name": state,
                "start_date": start_date
            }
            
            response = self._make_request("POST", "/forecast", json=payload)
            
            # Handle different response formats
            if isinstance(response, list):
                return response
            elif isinstance(response, dict) and "forecast" in response:
                return response["forecast"]
            elif isinstance(response, dict) and "data" in response:
                return response["data"]
            elif isinstance(response, dict) and "predictions" in response:
                return response["predictions"]
            else:
                logger.warning(f"Unexpected response format: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting forecast: {e}")
            raise Exception(f"Failed to get forecast: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if the API is accessible."""
        try:
            response = self._make_request("GET", "/")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Global API client instance
api_client = APIClient()

# Convenience functions to maintain compatibility with existing code
def get_available_states() -> List[str]:
    """Get list of available states."""
    return api_client.get_available_states()

def get_forecast(state: str, start_date: str) -> List[Dict[str, Any]]:
    """Get rainfall forecast for a state and date."""
    return api_client.get_forecast(state, start_date)

def check_api_health() -> bool:
    """Check if the API is healthy."""
    return api_client.health_check()
