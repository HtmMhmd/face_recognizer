from abc import ABC, abstractmethod
from typing import Any, Union, List, Tuple
import numpy as np

# Base class for all facial recognition models
class FacialRecognition(ABC):
    """
    Abstract base class for facial recognition models.
    All facial recognition models should inherit from this class.
    """
    model: Any
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int
    
    @abstractmethod
    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        Process an image through the facial recognition model to produce embeddings.
        
        Args:
            img: The preprocessed image as a numpy array
            
        Returns:
            Facial embedding as a numpy array
        """
        pass
