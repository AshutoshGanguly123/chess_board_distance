# chess_analysis/experiments.py
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import SVG, display

from chess_position import ChessPosition
from graph_construction import GraphConstructionFunctor
from evaluation import EvaluationFunctor
from algebraic_scores import (AlgebraicControlRefinementFunctor,
                              AlgebraicThreatRefinementFunctor,
                              AlgebraicStrategicRefinementFunctor,
                              AlgebraicHybridRefinementFunctor)
from utils import generate_random_positions, visualize_positions, normalize_eval

def run_experiments(num_positions=50, depth=10, num_sample_pairs=20,
                    stockfish_path="/opt/homebrew/bin/stockfish", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluator and graph functors
    evaluator = EvaluationFunctor(stockfish_path)
    control_functor = GraphConstructionFunctor("control")
    threat_functor = GraphConstructionFunctor("threat")
    strategic_functor = GraphConstructionFunctor("strategic")
    hybrid_functor = GraphConstructionFunctor("hybrid")

    # Algebraic refinement functors
    control_refinement = AlgebraicControlRefinementFunctor()
    threat_refinement = AlgebraicThreatRefinementFunctor()
    strategic_refinement = AlgebraicStrategicRefinementFunctor()
    hybrid_refinement = AlgebraicHybridRefinementFunctor()

    positions = generate_random_positions(num_positions, depth)
    if not positions:
        print("No valid positions generated.")
        return

    metrics = {
        "control_diff":   [],
        "threat_diff":    [],
        "strategic_diff": [],
        "hybrid_diff":    [],
        "eval_diff":      []
    }
    comparisons = []
    total_positions = len(positions)

    def compute_similarity_metrics(position1, position2):
        try:
            # Build graphs for both positions
            control_graph1   = control_functor(position1)
            threat_graph1    = threat_functor(position1)
            strategic_graph1 = strategic_functor(position1)
            hybrid_graph1    = hybrid_functor(position1)
            
            control_graph2   = control_functor(position2)
            threat_graph2    = threat_functor(position2)
            strategic_graph2 = strategic_functor(position2)
            hybrid_graph2    = hybrid_functor(position2)
            
            # Compute algebraic scores for each
            control_score1   = control_refinement(control_graph1)
            threat_score1    = threat_refinement(threat_graph1)
            strategic_score1 = strategic_refinement(strategic_graph1)
            hybrid_score1    = hybrid_refinement([control_graph1, threat_graph1, strategic_graph1])
            
            control_score2   = control_refinement(control_graph2)
            threat_score2    = threat_refinement(threat_graph2)
            strategic_score2 = strategic_refinement(strategic_graph2)
            hybrid_score2    = hybrid_refinement([control_graph2, threat_graph2, strategic_graph2])
            
            eval1 = normalize_eval(evaluator(position1))
            eval2 = normalize_eval(evaluator(position2))
            
            return {
                "control_diff":   abs(control_score1 - control_score2),
                "threat_diff":    abs(threat_score1 - threat_score2),
                "strategic_diff": abs(strategic_score1 - strategic_score2),
                "hybrid_diff":    abs(hybrid_score1 - hybrid_score2),
                "eval_diff":      abs(eval1 - eval2)
            }
        except Exception as e:
            print("Error computing similarity metrics for pair:", e)
            return None

    # Sample random pairs of positions
    for _ in range(num_sample_pairs):
        i, j = random.sample(range(total_positions), 2)
        metrics_diff = compute_similarity_metrics(positions[i], positions[j])
        if metrics_diff is not None:
            for key in metrics:
                metrics[key].append(metrics_diff[key])
            comparisons.append({
                "pos1": positions[i],
                "pos2": positions[j],
                "metrics": metrics_diff
            })

    if not comparisons:
        print("No valid position pairs for similarity metrics computation.")
        return

    avg_metrics = {key: np.mean(vals) if vals else 0.0 for key, vals in metrics.items()}
    print("Average Metric Differences Across Sample Pairs:")
    for key, avg in avg_metrics.items():
        print(f"{key}: {avg:.4f}")

    # Plot histograms for each metric difference
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    metric_names = ["control_diff", "threat_diff", "strategic_diff", "hybrid_diff", "eval_diff"]
    for ax, metric in zip(axs, metric_names):
        ax.hist(metrics[metric], bins=15, color='skyblue', edgecolor='black')
        ax.set_title(metric)
        ax.set_xlabel("Difference")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Save comparisons to separate folders
    for idx, comp in enumerate(comparisons):
        folder_name = os.path.join(output_dir, f"comparison_{idx+1}")
        os.makedirs(folder_name, exist_ok=True)
        
        svg1, svg2 = visualize_positions(comp["pos1"], comp["pos2"])
        with open(os.path.join(folder_name, "board1.svg"), "w") as f:
            f.write(svg1)
        with open(os.path.join(folder_name, "board2.svg"), "w") as f:
            f.write(svg2)
        
        with open(os.path.join(folder_name, "boards.txt"), "w") as f:
            f.write("Position 1 FEN:\n" + comp["pos1"].board.fen() + "\n\n")
            f.write("Position 2 FEN:\n" + comp["pos2"].board.fen() + "\n")
        
        with open(os.path.join(folder_name, "metrics.json"), "w") as f:
            json.dump(comp["metrics"], f, indent=4)

    # For interactive inspection: display one random comparison
    sample = random.choice(comparisons)
    print("\nVisualizing one random pair of boards:")
    svg1, svg2 = visualize_positions(sample["pos1"], sample["pos2"])
    html_str = f"""
    <div style="display: flex; gap: 20px;">
        <div>{svg1}</div>
        <div>{svg2}</div>
    </div>
    """
    display(SVG(html_str))
