
def print_results(results):
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
