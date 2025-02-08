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
from graph_construction import build_control_graph_side, build_threat_graph_side, build_strategic_graph_side, build_hybrid_graph_side

def compute_metrics_side(position, evaluator,
                         control_refinement, threat_refinement,
                         strategic_refinement, hybrid_refinement):
    """
    For a given ChessPosition, compute metrics for white and black separately.
    The strategic score is computed from the side-specific strategic graph.
    """
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
    
    # Engine evaluation: normalized eval is from White's perspective
    engine_val = normalize_eval(evaluator(position))
    white_eval = engine_val
    black_eval = -engine_val  # reverse for Black
    
    return {
        "white": {
            "control": white_control,
            "threat": white_threat,
            "strategic": white_strategic,
            "hybrid": white_hybrid,
            "eval": white_eval
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
    """
    Returns a dictionary mapping Queen’s Gambit variation names to a list of SAN moves.
    Each list corresponds to 15 full moves (30 half moves) of mainline theory.
    The variations provided below are among the most popular and theoretically precise.
    """
    return {
        "QGA (Queen's Gambit Accepted)": [
            "d4", "d5",
            "c4", "dxc4",
            "Nf3", "Nf6",
            "e3", "e6",
            "Bxc4", "c5",
            "O-O", "a6",
            "a4", "Nc6",
            "Qe2", "b5",
            "Bd3", "Bb7",
            "Nc3", "O-O",
            "Rd1", "Qc7",
            "e4", "cxd4",
            "exd5", "Nxd5",
            "Ne5", "Rfd8",
            "Bf4", "Re8"
        ],
        "QGD Orthodox Defense": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "Nf6",
            "Bg5", "Be7",
            "e3", "O-O",
            "Nf3", "h6",
            "Bh4", "b6",
            "cxd5", "exd5",
            "Bd3", "Bb7",
            "O-O", "Nbd7",
            "Rc1", "c5",
            "Qe2", "Rc8",
            "Rfd1", "Ne4",
            "Bb1", "Nxd4",
            "Bxe7", "Qxe7"
        ],
        "QGD Tarrasch Defense": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "c5",
            "cxd5", "exd5",
            "Nf3", "Nc6",
            "Bg5", "cxd4",
            "e3", "Nf6",
            "Bd3", "Bd6",
            "O-O", "O-O",
            "Rc1", "Re8",
            "Qe2", "Qa5",
            "Rfd1", "a6",
            "Bxf6", "Qxf6",
            "a3", "Be7",
            "Qd2"
        ],
        "QGD Lasker Defense": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "Nf6",
            "Bg5", "Be7",
            "e3", "O-O",
            "Nf3", "h6",
            "Bh4", "dxc4",
            "Bxc4", "c5",
            "O-O", "Nc6",
            "a4", "a6",
            "Qe2", "b5",
            "Rd1", "Bb7",
            "Ne5", "Qc7",
            "Nxc4", "Nxe5",
            "Bxe7", "Rxe7"
        ],
        "QGD Cambridge Springs": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "Nf6",
            "Bg5", "Bb4",
            "e3", "O-O",
            "Nf3", "c5",
            "Rc1", "Nc6",
            "Bd3", "cxd4",
            "exd4", "dxc4",
            "Bxc4", "b6",
            "O-O", "Bb7",
            "Qe2", "Rc8",
            "a3", "Ba5",
            "Rfd1", "Qe7",
            "Bxf6", "gxf6"
        ],
        "QGD Semi-Tarrasch": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "Nf6",
            "Bg5", "dxc4",
            "e4", "Bb4",
            "e5", "h6",
            "Bh4", "g5",
            "Nge2", "Nbd7",
            "O-O", "Bg7",
            "f4", "O-O",
            "Qe2", "c5",
            "Rfd1", "Qe7",
            "Rd3", "a6",
            "Bf2", "Nxc3",
            "bxc3"
        ],
        "QGD Exchange Variation": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "Nf6",
            "cxd5", "exd5",
            "Bg5", "c6",
            "e3", "Bb4",
            "Bd3", "O-O",
            "Nge2", "Re8",
            "O-O", "h6",
            "Bh4", "Nbd7",
            "Qe2", "Be7",
            "Rfd1", "a6",
            "Rac1", "Qc7",
            "a3", "Ba5",
            "Bxc2"
        ],
        "QGD Ragozin Defense": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "Nf6",
            "Bg5", "Bb4",
            "e3", "O-O",
            "Nf3", "c5",
            "Bd3", "dxc4",
            "Bxc4", "Nc6",
            "O-O", "Qe7",
            "a3", "Ba5",
            "Qc2", "Rd8",
            "Rfd1", "b6",
            "Rac1", "Bb7",
            "b4", "a6",
            "Na4"
        ],
        "QGD Chigorin Defense": [
            "d4", "d5",
            "c4", "Nc6",
            "Nc3", "Nf6",
            "cxd5", "Nxd5",
            "e4", "Nxc3",
            "bxc3", "e6",
            "Nf3", "Bd6",
            "Bd3", "O-O",
            "O-O", "Re8",
            "Qe2", "Bg4",
            "Rfd1", "a6",
            "Rac1", "Qe7",
            "f3", "Bh5",
            "Ne5", "Rxe5",
            "Qxh5"
        ],
        "QGD Noteboom Variation": [
            "d4", "d5",
            "c4", "e6",
            "Nc3", "Nf6",
            "Bg5", "Bb4",
            "e3", "O-O",
            "Nf3", "c5",
            "cxd5", "exd5",
            "Bd3", "Nc6",
            "O-O", "Re8",
            "Rc1", "a6",
            "Qe2", "b5",
            "Rac1", "Bd6",
            "Ne2", "Ne4",
            "Bxe7", "Qxe7",
            "f3"
        ]
    }


def run_openings_analysis(min_move=3, max_move=15,
                          stockfish_path="/opt/homebrew/bin/stockfish",
                          output_dir="queens_gambit_openings_experiment_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = EvaluationFunctor(stockfish_path)
    control_refinement   = AlgebraicControlRefinementFunctor()
    threat_refinement    = AlgebraicThreatRefinementFunctor()
    strategic_refinement = AlgebraicStrategicRefinementFunctor()
    hybrid_refinement    = AlgebraicHybridRefinementFunctor()
    
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
        
        white_avg = {}
        black_avg = {}
        if move_metrics:
            for key in ["control", "threat", "strategic", "hybrid", "eval"]:
                white_avg[key] = np.mean([m["white"][key] for m in move_metrics])
                black_avg[key] = np.mean([m["black"][key] for m in move_metrics])
        else:
            white_avg = {key: None for key in ["control", "threat", "strategic", "hybrid", "eval"]}
            black_avg = {key: None for key in ["control", "threat", "strategic", "hybrid", "eval"]}
        
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
        
        move_nums = [m["move_number"] for m in move_metrics]
        metrics_list = ["control", "threat", "strategic", "hybrid", "eval"]
        fig, axs = plt.subplots(nrows=len(metrics_list), ncols=1, figsize=(10, 20), sharex=True)
        for idx, metric in enumerate(metrics_list):
            white_values = [m["white"][metric] for m in move_metrics]
            black_values = [m["black"][metric] for m in move_metrics]
            axs[idx].plot(move_nums, white_values, marker="o", color='blue', label="White")
            axs[idx].plot(move_nums, black_values, marker="s", color='red', label="Black")
            axs[idx].set_title(metric.capitalize())
            axs[idx].grid(True)
            axs[idx].legend()
        axs[-1].set_xlabel("Move Number")
        fig.suptitle(f"Metrics Trend for {opening_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        trend_file = os.path.join(open_dir, f"{safe_name}_metrics_trend.png")
        plt.savefig(trend_file)
        plt.close()
        print(f"Saved analysis for {opening_name} in {open_dir}")
    
    def compute_similarity_matrix(opening_summaries):
        opening_names = list(opening_summaries.keys())
        feature_vectors = []
        for name in opening_names:
            avg = opening_summaries[name]
            if None not in avg.values():
                vec = [avg["control"], avg["threat"], avg["strategic"], avg["hybrid"], avg["eval"]]
                feature_vectors.append(vec)
            else:
                feature_vectors.append([0, 0, 0, 0, 0])
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
