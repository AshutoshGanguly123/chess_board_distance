# main.py
from experiments import run_experiments

if __name__ == "__main__":
    run_experiments(num_positions=15, depth=15, num_sample_pairs=40,
                    stockfish_path="/opt/homebrew/bin/stockfish", output_dir="chess_analysis/output")
