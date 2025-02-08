#!/usr/bin/env python3
import os
import json
import chess
import chess.pgn
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

def parse_and_analyze_games(pgn_filename, output_dir="dubov_experiments", min_move=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(pgn_filename) as pgn:
        game_index = 0
        evaluator = EvaluationFunctor("/opt/homebrew/bin/stockfish")
        control_refinement = AlgebraicControlRefinementFunctor()
        threat_refinement = AlgebraicThreatRefinementFunctor()
        strategic_refinement = AlgebraicStrategicRefinementFunctor()
        hybrid_refinement = AlgebraicHybridRefinementFunctor()
        
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            game_index += 1
            
            # Determine which side Dubov played
            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")
            if "Daniil Dubov" in white_player:
                dubov_color = "white"
                opponent = black_player
            elif "Daniil Dubov" in black_player:
                dubov_color = "black"
                opponent = white_player
            else:
                dubov_color = "unknown"
                opponent = "unknown"
            
            # Create a subfolder for this game (use event name or index)
            game_folder = os.path.join(output_dir, f"game_{game_index}")
            os.makedirs(game_folder, exist_ok=True)
            
            board = game.board()
            move_metrics = []
            move_number = 0
            # Process moves starting from move 'min_move'
            for move in game.mainline_moves():
                board.push(move)
                move_number += 1
                if move_number >= min_move:
                    pos = ChessPosition(board.fen())
                    metrics = compute_metrics_side(pos, evaluator,
                                                   control_refinement, threat_refinement,
                                                   strategic_refinement, hybrid_refinement)
                    metrics["move_number"] = move_number
                    move_metrics.append(metrics)
            
            # Compute average metrics over moves
            white_avg = {}
            black_avg = {}
            if move_metrics:
                for key in ["control", "threat", "strategic", "hybrid", "eval"]:
                    white_avg[key] = np.mean([m["white"][key] for m in move_metrics])
                    black_avg[key] = np.mean([m["black"][key] for m in move_metrics])
            else:
                white_avg = {key: None for key in ["control", "threat", "strategic", "hybrid", "eval"]}
                black_avg = {key: None for key in ["control", "threat", "strategic", "hybrid", "eval"]}
            
            summary = {
                "event": game.headers.get("Event", f"game_{game_index}"),
                "round": game.headers.get("Round", ""),
                "white": white_player,
                "black": black_player,
                "result": game.headers.get("Result", ""),
                "dubov_color": dubov_color,
                "opponent": opponent,
                "moves_used": [move.uci() for move in game.mainline_moves()],
                "average_metrics": {"white": white_avg, "black": black_avg},
                "move_by_move_metrics": move_metrics
            }
            summary_file = os.path.join(game_folder, "summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=4)
            
            # Plot move-by-move trends for each metric (White and Black)
            move_nums = [m["move_number"] for m in move_metrics]
            metrics_list = ["control", "threat", "strategic", "hybrid", "eval"]
            fig, axs = plt.subplots(nrows=len(metrics_list), ncols=1, figsize=(10, 20), sharex=True)
            for idx, metric in enumerate(metrics_list):
                white_values = [m["white"][metric] for m in move_metrics]
                black_values = [m["black"][metric] for m in move_metrics]
                axs[idx].plot(move_nums, white_values, marker="o", color='blue', label="White")
                axs[idx].plot(move_nums, black_values, marker="s", color='red', label="Black")
                axs[idx].set_title(f"{metric.capitalize()} (Dubov as {dubov_color})")
                axs[idx].grid(True)
                axs[idx].legend()
            axs[-1].set_xlabel("Move Number")
            fig.suptitle(f"Metrics Trend for Game {game_index} ({dubov_color.capitalize()} Dubov vs {opponent})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            trend_file = os.path.join(game_folder, "metrics_trend.png")
            plt.savefig(trend_file)
            plt.close()
            print(f"Processed game {game_index}: Dubov as {dubov_color} vs {opponent}")

def main():
    pgn_file = "dubov_games.pgn"  # Replace with the path to your PGN file containing the games
    parse_and_analyze_games(pgn_file, output_dir="dubov_experiments", min_move=3)

if __name__ == "__main__":
    main()
