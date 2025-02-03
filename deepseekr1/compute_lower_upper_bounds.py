import json
import random
from main import generate_random_positions, compute_similarity_metrics

def calculate_bounds(num_positions=50, depth=15, num_sample_pairs=1000):
    """
    Generate a set of random chess positions and then compute the similarity metrics
    for num_sample_pairs random pairs. Record the minimum and maximum observed values
    for each metric.
    
    Returns a dictionary with keys:
       "control_diff", "threat_diff", "strategic_diff", "hybrid_diff", "eval_diff"
    Each key maps to a dict with "min" and "max" values.
    """
    # Generate a pool of random positions (filtered as in your main code)
    positions = generate_random_positions(num_positions, depth)
    if not positions:
        raise ValueError("No valid positions generated.")
    
    # Define the metric keys we are interested in
    metrics_keys = ["control_diff", "threat_diff", "strategic_diff", "hybrid_diff", "eval_diff"]
    
    # Initialize bounds dictionary
    bounds = {k: {"min": float("inf"), "max": float("-inf")} for k in metrics_keys}
    total_positions = len(positions)
    
    # Sample pairs randomly and update the min/max bounds for each metric
    for _ in range(num_sample_pairs):
        i, j = random.sample(range(total_positions), 2)
        metrics_diff = compute_similarity_metrics(positions[i], positions[j])
        if metrics_diff is None:
            continue  # Skip invalid comparisons
        for key in metrics_keys:
            value = metrics_diff[key]
            if value < bounds[key]["min"]:
                bounds[key]["min"] = value
            if value > bounds[key]["max"]:
                bounds[key]["max"] = value
                
    # In case no valid value was updated, replace infinities with None
    for key in bounds:
        if bounds[key]["min"] == float("inf"):
            bounds[key]["min"] = None
        if bounds[key]["max"] == float("-inf"):
            bounds[key]["max"] = None
            
    return bounds

if __name__ == "__main__":
    bounds = calculate_bounds(num_positions=50, depth=15, num_sample_pairs=1000)
    # Save the calculated bounds to a JSON file for later use
    with open("metric_bounds.json", "w") as f:
        json.dump(bounds, f, indent=4)
    print("Saved metric bounds to metric_bounds.json:")
    print(json.dumps(bounds, indent=4))
