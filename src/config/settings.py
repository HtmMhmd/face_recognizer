"""
Configuration settings for the Face Recognition system.
This module loads settings from the config.yaml file and provides
access to these settings throughout the application.
"""

import os
import yaml
from typing import Any, Dict, List, Optional, Union

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default config path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

class Settings:
    """
    Load and manage application settings from the config.yaml file.
    Provides convenient access to configuration values with dot notation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize settings from the config.yaml file.
        
        Args:
            config_path: Path to the configuration file (defaults to src/config/config.yaml)
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        
        # Quick access to common settings
        self.app = self._get_section("app")
        self.api = self._get_section("api")
        self.camera = self._get_section("camera")
        self.detection = self._get_section("detection")
        self.recognition = self._get_section("recognition")
        self.verification = self._get_section("verification")
        self.drowsiness = self._get_section("drowsiness")
        self.database = self._get_section("database")
        self.alignment = self._get_section("alignment")
        self.image = self._get_section("image")
        self.output = self._get_section("output")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from the YAML file."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {str(e)}")
            # Return empty dictionary if the file can't be read
            return {}
    
    def _get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific section from the config."""
        return self.config.get(section, {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot notation key (e.g. "detection.mediapipe.min_detection_confidence")
            default: Default value if key not found
            
        Returns:
            The configuration value or default
        """
        parts = key.split(".")
        config = self.config
        
        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default
        
        return config
    
    def get_path(self, path_key: str) -> str:
        """
        Get a file path from the configuration, joining it with the base directory.
        
        Args:
            path_key: The key for the path in dot notation
            
        Returns:
            The absolute path
        """
        path = self.get(path_key)
        if not path:
            return ""
            
        # If it's already an absolute path, return it directly
        if os.path.isabs(path):
            return path
            
        # Otherwise join with the base directory
        return os.path.join(BASE_DIR, path)
    
    def reload(self):
        """Reload the configuration from the file."""
        self.config = self._load_config()
        
        # Update quick access attributes
        self.app = self._get_section("app")
        self.api = self._get_section("api")
        self.camera = self._get_section("camera")
        self.detection = self._get_section("detection")
        self.recognition = self._get_section("recognition")
        self.verification = self._get_section("verification")
        self.drowsiness = self._get_section("drowsiness")
        self.database = self._get_section("database")
        self.alignment = self._get_section("alignment")
        self.image = self._get_section("image")
        self.output = self._get_section("output")

# Create a global instance of Settings
settings = Settings()

# For convenience, export all sections as global variables
app_settings = settings.app
api_settings = settings.api
camera_settings = settings.camera
detection_settings = settings.detection
recognition_settings = settings.recognition
verification_settings = settings.verification
drowsiness_settings = settings.drowsiness
database_settings = settings.database
alignment_settings = settings.alignment
image_settings = settings.image
output_settings = settings.output

def get_setting(key: str, default: Any = None) -> Any:
    """
    Convenience function to get a setting value by key.
    
    Args:
        key: The key in dot notation
        default: Default value if key not found
        
    Returns:
        The setting value or default
    """
    return settings.get(key, default)

def get_path(path_key: str) -> str:
    """
    Convenience function to get a path from the settings.
    
    Args:
        path_key: The key for the path in dot notation
        
    Returns:
        The absolute path
    """
    return settings.get_path(path_key)
