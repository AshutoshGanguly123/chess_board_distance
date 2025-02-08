#!/usr/bin/env python3
import os
import json
import chess
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from chess_position import ChessPosition
from evaluation import EvaluationFunctor
from algebraic_scores import (AlgebraicControlRefinementFunctor,
                              AlgebraicThreatRefinementFunctor,
                              AlgebraicStrategicRefinementFunctor,
                              AlgebraicHybridRefinementFunctor)
from utils import normalize_eval
from graph_construction import (build_control_graph_side,
                                build_threat_graph_side,
                                build_strategic_graph_side,
                                build_hybrid_graph_side)

def compute_metrics_side(position, evaluator,
                         control_refinement, threat_refinement,
                         strategic_refinement, hybrid_refinement):
    # White side
    white_control = control_refinement(build_control_graph_side(position, chess.WHITE))
    white_threat = threat_refinement(build_threat_graph_side(position, chess.WHITE))
    white_strategic = strategic_refinement(build_strategic_graph_side(position, chess.WHITE), position, chess.WHITE)
    white_hybrid = hybrid_refinement([
        build_control_graph_side(position, chess.WHITE),
        build_threat_graph_side(position, chess.WHITE),
        build_strategic_graph_side(position, chess.WHITE)
    ])
    
    # Black side
    black_control = control_refinement(build_control_graph_side(position, chess.BLACK))
    black_threat = threat_refinement(build_threat_graph_side(position, chess.BLACK))
    black_strategic = strategic_refinement(build_strategic_graph_side(position, chess.BLACK), position, chess.BLACK)
    black_hybrid = hybrid_refinement([
        build_control_graph_side(position, chess.BLACK),
        build_threat_graph_side(position, chess.BLACK),
        build_strategic_graph_side(position, chess.BLACK)
    ])
    
    engine_val = normalize_eval(evaluator(position))
    white_eval = engine_val
    black_eval = -engine_val
    
    return {
        "white": {
            "control": white_control,     # dict with "fiedler" and "fiedler_centrality"
            "threat": white_threat,         # dict with "spectral_radius" and "threat_centrality"
            "strategic": white_strategic,   # dict with 5 sub-metrics
            "hybrid": white_hybrid,         # float
            "eval": white_eval            # float
        },
        "black": {
            "control": black_control,
            "threat": black_threat,
            "strategic": black_strategic,
            "hybrid": black_hybrid,
            "eval": black_eval
        }
    }

def fetch_top_openings():
    return {
 "Ruy Lopez": [
            "e4", "e5",
            "Nf3", "Nc6",
            "Bb5", "a6",
            "Ba4", "Nf6",
            "O-O", "Be7",
            "Re1", "b5",
            "Bb3", "d6",
            "c3", "O-O",
            "h3", "Nb8",
            "d4", "Nbd7",
            "Nbd2", "Bb7",
            "Bc2", "Re8",
            "Nf1", "Bf8",
            "Ng3", "g6",
            "a4", "exd4"
        ],
        "Sicilian Defense": [
            "e4", "c5",
            "Nf3", "d6",
            "d4", "cxd4",
            "Nxd4", "Nf6",
            "Nc3", "a6",
            "Be3", "e5",
            "Nb3", "Be6",
            "f3", "Be7",
            "Qd2", "O-O",
            "O-O-O", "Nbd7",
            "g4", "b5",
            "g5", "b4",
            "Na4", "Rc8",
            "Kb1", "Nc5",
            "Bc4", "Nfd7"
        ],
        "French Defense": [
            "e4", "e6",
            "d4", "d5",
            "Nc3", "Nf6",
            "Bg5", "Bb4",
            "e5", "h6",
            "Bh4", "O-O",
            "Nge2", "c5",
            "a3", "Bxc3+",
            "bxc3", "Nc6",
            "Qg4", "g6",
            "Bd3", "Ba6",
            "O-O", "Qxd3",
            "Rfe1", "Qxc2",
            "Ng3", "cxd4",
            "cxd4", "Nxd4"
        ],
        "Caro-Kann Defense": [
            "e4", "c6",
            "d4", "d5",
            "Nc3", "dxe4",
            "Nxe4", "Nd7",
            "Ng5", "Ngf6",
            "Bd3", "h6",
            "Nge4", "e6",
            "Nxf6+", "Nxf6",
            "O-O", "Bd6",
            "Re1", "O-O",
            "c3", "Qc7",
            "Bg5", "Rad8",
            "Qd3", "c5",
            "dxc5", "Bxc5",
            "Rad1", "Rd7"
        ],
        "Queen's Gambit": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "Nf6",
            "Bg5", "Bb4",
            "e3", "O-O",
            "Nf3", "c5",
            "cxd5", "exd5",
            "Bd3", "Nc6",
            "O-O", "Re8",
            "Rc1", "b6",
            "a3", "Ba6",
            "Qe2", "Bxd3",
            "Qxd3", "Qb8",
            "Rfd1", "Rac8",
            "Bf1", "Rxc1"
        ],
        "King's Indian Defense": [
            "d4", "Nf6",
            "c4", "g6",
            "Nc3", "Bg7",
            "e4", "d6",
            "Nf3", "O-O",
            "Be2", "e5",
            "O-O", "Nc6",
            "d5", "Ne7",
            "b4", "a5",
            "Ba3", "axb4",
            "Bxb4", "f5",
            "a4", "Nf5",
            "Nd2", "Nxd4",
            "Qc2", "Qe8",
            "Rab1", "Rf6"
        ],
        "Nimzo-Indian Defense": [
            "d4", "Nf6",
            "c4", "e6",
            "Nc3", "Bb4",
            "Qc2", "O-O",
            "a3", "Bxc3+",
            "Qxc3", "c5",
            "e4", "d5",
            "Bd3", "Nc6",
            "Nf3", "dxc4",
            "Bxc4", "a6",
            "O-O", "b5",
            "Be2", "Bb7",
            "Rfd1", "Qc7",
            "Rd3", "Rad8",
            "Ne5", "Rfe8"
        ],
        "English Opening": [
            "c4", "e5",
            "Nc3", "Nf6",
            "g3", "Nc6",
            "Bg2", "d6",
            "d4", "exd4",
            "Nxd4", "Be7",
            "Nf3", "O-O",
            "O-O", "a6",
            "a4", "Re8",
            "Qc2", "b6",
            "Rd1", "Bb7",
            "b3", "Qc7",
            "Bb2", "Rab8",
            "Rfe1", "Nd4",
            "e3", "Nxf3+"
        ],
        "Italian Game": [
            "e4", "e5",
            "Nf3", "Nc6",
            "Bc4", "Bc5",
            "c3", "Nf6",
            "d4", "exd4",
            "cxd4", "Bb4+",
            "Nc3", "d5",
            "exd5", "Nxd5",
            "O-O", "O-O",
            "Re1", "Be6",
            "Bg5", "f6",
            "Bd3", "Ne7",
            "Qb3", "c6",
            "Nxd5", "cxd5",
            "Rad1", "Re8"
        ],
        "Scandinavian Defense": [
            "e4", "d5",
            "exd5", "Qxd5",
            "Nc3", "Qa5",
            "d4", "Nf6",
            "Nf3", "c6",
            "Bc4", "Bf5",
            "O-O", "e6",
            "Re1", "Bb4",
            "Bd2", "O-O",
            "a3", "Bxc3",
            "Bxc3", "Nbd7",
            "Qd3", "Rad8",
            "Be3", "Qc7",
            "Rxe6", "Nxe4",
            "Qe2", "Rxd4"
        ],
        "Pirc Defense": [
            "e4", "d6",
            "d4", "Nf6",
            "Nc3", "g6",
            "Nf3", "Bg7",
            "Be2", "O-O",
            "O-O", "c6",
            "a4", "b6",
            "Be3", "Bb7",
            "Qd2", "Nbd7",
            "Rad1", "Re8",
            "b3", "Qc7",
            "Rfe1", "Rab8",
            "Bb1", "a6",
            "h3", "Nc5",
            "Bc1", "Rf8"
        ],
        "Modern Defense": [
            "e4", "g6",
            "d4", "Bg7",
            "Nc3", "d6",
            "f4", "Nf6",
            "Nf3", "O-O",
            "Be2", "c6",
            "O-O", "Qc7",
            "a4", "b6",
            "Kh1", "Bb7",
            "Bd3", "a6",
            "Qe1", "Rfe8",
            "Qh4", "exf4",
            "Bxf4", "Nd7",
            "Rf1", "Nf8",
            "Rf2", "Rf8"
        ],
        "Dutch Defense": [
            "d4", "f5",
            "c4", "Nf6",
            "Nc3", "e6",
            "Nf3", "d5",
            "Bg5", "Bb4",
            "e3", "O-O",
            "Bd3", "c5",
            "O-O", "Nc6",
            "a3", "Bxc3",
            "bxc3", "dxc4",
            "Qe2", "Qe7",
            "Rfd1", "Rd8",
            "Ne5", "Rxd4",
            "Bf4", "Nd5",
            "Qc2", "e5"
        ],
        "Vienna Game": [
            "e4", "e5",
            "Nc3", "Nf6",
            "f4", "d5",
            "fxe5", "Nxe4",
            "Nf3", "Bc5",
            "Bb5+", "c6",
            "d4", "Bb6",
            "O-O", "O-O",
            "Qe2", "Qe7",
            "Bxc6", "bxc6",
            "dxe5", "Re8",
            "Qe2", "Qxe7",
            "Rxe7", "Nc3",
            "Rxe7", "Nxa2",
            "Rxd5", "Rxd5"
        ],
        "Scotch Game": [
            "e4", "e5",
            "Nf3", "Nc6",
            "d4", "exd4",
            "Nxd4", "Nf6",
            "Nc3", "Bb4",
            "Nxc6", "bxc6",
            "Bd3", "d5",
            "O-O", "O-O",
            "Bg5", "Re8",
            "Re1", "h6",
            "Bh4", "c5",
            "exd5", "cxd5",
            "Qf3", "Be7",
            "Rad1", "Qd7",
            "Bxf6", "Qxf6"
        ],
        "Closed Sicilian": [
            "e4", "c5",
            "Nc3", "Nc6",
            "g3", "g6",
            "Bg2", "Bg7",
            "d3", "d6",
            "Nge2", "Nf6",
            "O-O", "O-O",
            "h3", "Rb8",
            "Rb1", "Bd7",
            "a3", "b5"
        ]

        # Other openings omitted for brevity.
    }

def average_metric(metric_list):
    """
    Averages a list of metric values.
    If the first element is a dictionary, then averages each sub-key.
    """
    if isinstance(metric_list[0], dict):
        keys = metric_list[0].keys()
        avg = {}
        for k in keys:
            avg[k] = np.mean([m[k] for m in metric_list])
        return avg
    else:
        return np.mean(metric_list)

def run_openings_analysis(min_move=3, max_move=15,
                          stockfish_path="/opt/homebrew/bin/stockfish",
                          output_dir="openings_experiment_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = EvaluationFunctor(stockfish_path)
    control_refinement = AlgebraicControlRefinementFunctor()
    threat_refinement = AlgebraicThreatRefinementFunctor()
    strategic_refinement = AlgebraicStrategicRefinementFunctor()
    hybrid_refinement = AlgebraicHybridRefinementFunctor()
    
    openings = fetch_top_openings()
    white_opening_summaries = {}
    black_opening_summaries = {}
    
    for opening_name, moves in openings.items():
        print(f"Analyzing opening: {opening_name}")
        safe_name = opening_name.replace(" ", "_")
        open_dir = os.path.join(output_dir, safe_name)
        os.makedirs(open_dir, exist_ok=True)
        
        move_metrics = []
        last_move = min(max_move, len(moves))
        if last_move < min_move:
            print(f"Opening '{opening_name}' does not have enough moves. Skipping.")
            continue
        
        board = chess.Board()
        positions = []
        for idx, move in enumerate(moves[:last_move], start=1):
            try:
                board.push_san(move)
            except Exception as e:
                print(f"Error applying move {move} in {opening_name}: {e}")
                break
            if idx >= min_move:
                positions.append(ChessPosition(board.fen()))
        
        for i, pos in enumerate(positions, start=min_move):
            metrics = compute_metrics_side(pos, evaluator,
                                           control_refinement, threat_refinement,
                                           strategic_refinement, hybrid_refinement)
            metrics["move_number"] = i
            move_metrics.append(metrics)
        
        # Compute average metrics per opening.
        white_avg = {}
        black_avg = {}
        # For each key, if its value is a dict, average each sub-key.
        for key in ["control", "threat", "hybrid", "eval"]:
            white_avg[key] = average_metric([m["white"][key] for m in move_metrics])
            black_avg[key] = average_metric([m["black"][key] for m in move_metrics])
        # For strategic metrics (which are dictionaries of 5 values):
        white_avg["strategic"] = average_metric([m["white"]["strategic"] for m in move_metrics])
        black_avg["strategic"] = average_metric([m["black"]["strategic"] for m in move_metrics])
        
        white_opening_summaries[opening_name] = white_avg
        black_opening_summaries[opening_name] = black_avg
        
        summary = {
            "opening": opening_name,
            "moves_used": moves[:last_move],
            "average_metrics": {"white": white_avg, "black": black_avg},
            "move_by_move_metrics": move_metrics
        }
        summary_file = os.path.join(open_dir, f"{safe_name}_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)
        
        # Plot move-by-move trends.
        # For non-dictionary keys (hybrid, eval), plot as before.
        # For dictionary keys (control, threat, strategic), plot each sub-key in its own subplot.
        move_nums = [m["move_number"] for m in move_metrics]
        non_dict_keys = ["hybrid", "eval"]
        dict_keys = ["control", "threat", "strategic"]
        # Count total subplots: for each dict key, count subkeys; plus one subplot per non-dict key.
        total_plots = 0
        subplots_info = []  # list of (key, subkey) or (key,None) for non-dict keys
        for key in dict_keys:
            sample = move_metrics[0]["white"][key]
            if isinstance(sample, dict):
                for subkey in sample.keys():
                    subplots_info.append((key, subkey))
                    total_plots += 1
            else:
                subplots_info.append((key, None))
                total_plots += 1
        for key in non_dict_keys:
            subplots_info.append((key, None))
            total_plots += 1
        
        fig, axs = plt.subplots(nrows=total_plots, ncols=1, figsize=(10, 3*total_plots), sharex=True)
        if total_plots == 1:
            axs = [axs]
        ax_index = 0
        for key, subkey in subplots_info:
            if subkey is None:
                white_values = [m["white"][key] for m in move_metrics]
                black_values = [m["black"][key] for m in move_metrics]
                title = key.capitalize()
            else:
                white_values = [m["white"][key][subkey] for m in move_metrics]
                black_values = [m["black"][key][subkey] for m in move_metrics]
                title = f"{key.capitalize()} - {subkey.replace('_', ' ').capitalize()}"
            axs[ax_index].plot(move_nums, white_values, marker="o", color="blue", label="White")
            axs[ax_index].plot(move_nums, black_values, marker="s", color="red", label="Black")
            axs[ax_index].set_title(title)
            axs[ax_index].set_xticks(move_nums)
            axs[ax_index].set_xticklabels([str(m) for m in move_nums], rotation=45)
            axs[ax_index].grid(True)
            axs[ax_index].legend()
            ax_index += 1
        axs[-1].set_xlabel("Move Number")
        fig.suptitle(f"Metrics Trend for {opening_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        trend_file = os.path.join(open_dir, f"{safe_name}_metrics_trend.png")
        plt.savefig(trend_file)
        plt.close()
        print(f"Saved analysis for {opening_name} in {open_dir}")
    
    # Similarity matrices (flatten the metrics into one vector per opening)
    def compute_similarity_matrix(opening_summaries):
        opening_names = list(opening_summaries.keys())
        feature_vectors = []
        for name in opening_names:
            avg = opening_summaries[name]
            if avg["control"] is not None and avg["threat"] is not None and avg["strategic"] is not None:
                vec = []
                # For control, threat, strategic: flatten their dictionaries.
                for key in ["control", "threat", "strategic"]:
                    for subkey in sorted(avg[key].keys()):
                        vec.append(avg[key][subkey])
                # Append non-dictionary metrics.
                for key in ["hybrid", "eval"]:
                    vec.append(avg[key])
                feature_vectors.append(vec)
            else:
                feature_vectors.append([0]* (3*2 + 2))  # 3 dicts with 2 subkeys each assumed + 2
        num_openings = len(opening_names)
        sim_matrix = np.zeros((num_openings, num_openings))
        for i in range(num_openings):
            for j in range(num_openings):
                diff = np.array(feature_vectors[i]) - np.array(feature_vectors[j])
                sim_matrix[i, j] = sqrt(np.sum(diff**2))
        return opening_names, sim_matrix

    white_names, white_sim = compute_similarity_matrix(white_opening_summaries)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(white_sim, cmap="viridis")
    plt.colorbar(im, label="Euclidean Distance")
    plt.xticks(range(len(white_names)), [name.replace(" ", "\n") for name in white_names], rotation=45, ha="right")
    plt.yticks(range(len(white_names)), [name.replace(" ", "\n") for name in white_names])
    plt.title("White Opening Similarity Matrix (Lower = More Similar)")
    plt.tight_layout()
    white_sim_file = os.path.join(output_dir, "white_openings_similarity_matrix.png")
    plt.savefig(white_sim_file)
    plt.close()
    print(f"Saved white similarity matrix in {white_sim_file}")
    
    black_names, black_sim = compute_similarity_matrix(black_opening_summaries)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(black_sim, cmap="viridis")
    plt.colorbar(im, label="Euclidean Distance")
    plt.xticks(range(len(black_names)), [name.replace(" ", "\n") for name in black_names], rotation=45, ha="right")
    plt.yticks(range(len(black_names)), [name.replace(" ", "\n") for name in black_names])
    plt.title("Black Opening Similarity Matrix (Lower = More Similar)")
    plt.tight_layout()
    black_sim_file = os.path.join(output_dir, "black_openings_similarity_matrix.png")
    plt.savefig(black_sim_file)
    plt.close()
    print(f"Saved black similarity matrix in {black_sim_file}")

if __name__ == "__main__":
    run_openings_analysis()
