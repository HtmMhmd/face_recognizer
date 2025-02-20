import numpy as np

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

    def print_results2(self, results):
        metrices = ''
        verified = ''
        distance = ''
        threash  = ''
        for metric, result in results.items():
            metrices = metrices + f"{metric}| "
            verified = verified + f"{result['verified']}| "
            distance = distance + f"{result['distance']:.2f}| "
            threash  = threash  + f"{result['threshold']}| "
        print('__________________________')
        print(f" | Metric   : {metrices}")
        print(f" | Verified : {verified}")
        print(f" | Distance : {distance}")
        print(f" | Threshold: {threash}")
        print('__________________________')
    def print_results(self, results):
        """Prints verification results in a square-like format with consistent alignment
        and limits distance/threshold to two decimal places.

        Args:
            results (dict): A dictionary where keys are metric names ('euclidean', 'cosine', etc.)
                            and values are dictionaries with 'verified' (bool), 'distance' (float),
                            and 'threshold' (float).
        """

        if not isinstance(results, dict):
            print("Error: 'results' must be a dictionary.")
            return

        metrics = list(results.keys())
        if not metrics:
            print("Error: 'results' dictionary is empty.")
            return

        # Determine max metric name length for alignment
        max_metric_len = max(len(metric) for metric in metrics)

        # Format string for header row
        header_format = f"| {{:^{max_metric_len}}} ".format  # Center-align headers

        # Calculate total width based on metric length
        total_width = (max_metric_len + 3) * len(metrics) + 1  # +3 for ' | ' after each metric + leading '|'

        # Print top border
        print("+" + "-" * (total_width - 2) + "+")

        # Print metric names (header row)
        header_row = "|"
        for metric in metrics:
            header_row += header_format(metric) + "|"
        print(header_row)

        # Print separator line after header
        print("+" + "-" * (total_width - 2) + "+")

        # Define values to display
        values = ['Verified', 'Distance', 'Threshold']

        for value_type in values:
            row = "| "
            for metric in metrics:
                result = results.get(metric)

                if not isinstance(result, dict):
                    val_str = "N/A"  # Handle missing or invalid results
                else:
                    val = result.get(value_type.lower())  # Get value, case-insensitive

                    if val is None:
                        val_str = "N/A" # Handle cases where key is not present
                    elif value_type == 'Verified':
                        val_str = str(val)  # Print boolean as string
                    elif value_type in ('Distance', 'Threshold'):
                        try:
                            val_str = f"{float(val):.2f}"  # Format to 2 decimal places, converting to float first
                        except (ValueError, TypeError):
                            val_str = "N/A" #if something goes wrong with conversion
                    else:
                        val_str = str(val)   # Default to string conversion


                row += f" {{:^{max_metric_len}}} |".format(val_str)
            print(row)

        # Print bottom border
        print("+" + "-" * (total_width - 2) + "+")


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
            self.print_results(results)

        return results

# Example usage
if __name__ == '__main__':
    verifier = FaceVerifier()
    embedding1 = np.random.rand(512)  # Example embedding 1
    embedding2 = np.random.rand(512)  # Example embedding 2
    result_all_metrics = verifier.verify_faces(embedding1, embedding2, metric=None, verbose=True)