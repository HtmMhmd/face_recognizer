import numpy as np
from src.config import verification_settings

class FaceVerifier:
    def __init__(self):
        # Use thresholds from config
        self.thresholds = verification_settings.get("thresholds", {}).get("default", {})
        if not self.thresholds:
            # Fallback defaults if config is empty
            self.thresholds = {
                "cosine": 0.40,
                "euclidean": 10,
                "euclidean_l2": 0.80
            }

    def find_euclidean_distance(self, embedding1, embedding2):
        distance_vector = np.square(embedding1 - embedding2)
        return np.sqrt(distance_vector.sum())

    def find_cosine_similarity(self, embedding1, embedding2):
        return 1- (np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def find_euclidean_l2_distance(self, embedding1, embedding2):
        l2_embedding1 = embedding1 / np.linalg.norm(embedding1)
        l2_embedding2 = embedding2 / np.linalg.norm(embedding2)
        return np.linalg.norm(l2_embedding1 - l2_embedding2)

    def find_threshold(self, model_name: str, distance_metric: str) -> float:
        # Use thresholds from config
        base_threshold = verification_settings.get("thresholds", {}).get("default", {})
        if not base_threshold:
            base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

        # Get model-specific thresholds from config
        thresholds = verification_settings.get("thresholds", {})
        
        # Get threshold for the specific model and metric
        threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)
        return threshold

    def verify_faces(self, embedding1, embedding2, model_name='Facenet512', metric=None, verbose=False):
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)
        if embedding1.shape != (512,) or embedding2.shape != (512,):
            raise ValueError("Embeddings must be of shape (512,)")

        results = {}

        # Use metrics from config if not specified
        if metric is None:
            metrics = verification_settings.get("metrics", ['euclidean', 'cosine', 'euclidean_l2'])
        else:
            metrics = [metric]

        for metric in metrics:
            threshold = self.find_threshold(model_name, metric)
            if metric == 'euclidean':
                distance = self.find_euclidean_distance(embedding1, embedding2)
                verified = distance <= threshold
            elif metric == 'cosine':
                similarity = self.find_cosine_similarity(embedding1, embedding2)
                verified = similarity <= threshold
                distance = similarity  # For consistency with threshold comparison
            elif metric == 'euclidean_l2':
                distance = self.find_euclidean_l2_distance(embedding1, embedding2)
                verified = distance <= threshold
            else:
                raise ValueError("Invalid metric. Choose 'euclidean', 'cosine', or 'euclidean_l2'.")

            results[metric] = {
                "verified": verified,
                "distance": distance,
                "threshold": threshold,
                "metric": metric,
            }

        # Check if we require all metrics to pass
        require_all = verification_settings.get("require_all_metrics", True)
        results["all_verified"] = all(r["verified"] for r in results.values()) if require_all else any(r["verified"] for r in results.values())

        # Print results if verbose is True
        if verbose:
            from .utils import print_results
            print_results(results)

        return results