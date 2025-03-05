import numpy as np
from .verify_utilis import print_results

class FaceVerifier:
    def __init__(self):
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
        base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

        thresholds = {
            "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        }

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

        metrics = ['euclidean', 'cosine', 'euclidean_l2'] if metric is None else [metric]

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

        # Print results if verbose is True
        if verbose:
            print_results(results)

        return results

# Example usage
if __name__ == '__main__':
    verifier = FaceVerifier()
    embedding1 = np.random.rand(512)  # Example embedding 1
    embedding2 = np.random.rand(512)  # Example embedding 2
    result_all_metrics = verifier.verify_faces(embedding1, embedding2, metric=None, verbose=True)