# Install dependencies (uncomment these lines if running in a notebook)
# !pip install python-chess networkx stockfish numpy scipy matplotlib

import chess
import chess.svg
import networkx as nx
import numpy as np
from stockfish import Stockfish
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
import random
import matplotlib.pyplot as plt
from IPython.display import SVG, display
import os
import json

# ---------------------------
# Chess Position and Morphism
# ---------------------------
class ChessPosition:
    def __init__(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen)
    
    def morphism(self, move):
        """Return a new ChessPosition after making a legal move."""
        new_board = self.board.copy()
        new_board.push(move)
        return ChessPosition(new_board.fen())
    
    def __eq__(self, other):
        return self.board.fen() == other.board.fen()

# ---------------------------
# Graph Construction Functor
# ---------------------------
class GraphConstructionFunctor:
    def __init__(self, graph_type="control"):
        self.graph_type = graph_type.lower()
    
    def __call__(self, position):
        if self.graph_type == "control":
            return self._build_control_graph(position)
        elif self.graph_type == "threat":
            return self._build_threat_graph(position)
        elif self.graph_type == "strategic":
            return self._build_strategic_graph(position)
        elif self.graph_type == "hybrid":
            return self._build_hybrid_graph(position)
        else:
            raise ValueError("Unsupported graph type")

    def _build_control_graph(self, position):
        G = nx.Graph()
        board = position.board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                G.add_node(square, piece=piece.symbol())
                for target in board.attacks(square):
                    if board.piece_at(target):
                        G.add_edge(square, target)
        return G

    def _build_threat_graph(self, position):
        G = nx.DiGraph()
        board = position.board
        for square in chess.SQUARES:
            attacker = board.piece_at(square)
            if attacker:
                for victim_sq in board.attacks(square):
                    victim = board.piece_at(victim_sq)
                    if victim and attacker.color != victim.color:
                        G.add_edge(square, victim_sq)
        return G

    def _build_strategic_graph(self, position):
        G = nx.Graph()  # Using undirected for strategic measures
        board = position.board
        central_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for sq in central_squares:
            if board.piece_at(sq):
                G.add_node(sq, role="central")
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is not None:
                G.add_node(king_sq, role="king")
        nodes = list(G.nodes())
        for i, sq1 in enumerate(nodes):
            for sq2 in nodes[i+1:]:
                if (abs(chess.square_file(sq1) - chess.square_file(sq2)) <= 1 and
                    abs(chess.square_rank(sq1) - chess.square_rank(sq2)) <= 1):
                    G.add_edge(sq1, sq2)
        return G

    def _build_hybrid_graph(self, position):
        control_graph = self._build_control_graph(position)
        threat_graph = self._build_threat_graph(position).to_undirected()  # convert for consistency
        strategic_graph = self._build_strategic_graph(position)
        if (control_graph.number_of_nodes() == 0 and 
            threat_graph.number_of_nodes() == 0 and 
            strategic_graph.number_of_nodes() == 0):
            return nx.Graph()
        try:
            G = nx.compose_all([control_graph, threat_graph, strategic_graph])
        except Exception as e:
            print("Error composing graphs:", e)
            G = nx.Graph()
        return G

# ---------------------------
# Evaluation Functor using Stockfish
# ---------------------------
class EvaluationFunctor:
    def __init__(self, stockfish_path):
        self.evaluator = Stockfish(stockfish_path)
    
    def __call__(self, position):
        try:
            self.evaluator.set_fen_position(position.board.fen())
            eval_result = self.evaluator.get_evaluation()
        except Exception as e:
            print("Error evaluating position:", e)
            eval_result = {'type': 'cp', 'value': 0}
        return eval_result

# ---------------------------
# Algebraic Score Functions with Edge Case Handling
# ---------------------------
def algebraic_control_score(graph):
    if graph.number_of_nodes() == 0:
        return 0.0
    try:
        L = nx.laplacian_matrix(graph).toarray().astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        if len(eigenvalues) > 1:
            lambda2 = eigenvalues[1]
            fiedler_vector = eigenvectors[:, 1]
        else:
            lambda2 = eigenvalues[0]
            fiedler_vector = eigenvectors[:, 0]
        centrality = np.max(np.abs(fiedler_vector))
        return lambda2 + centrality
    except Exception as e:
        print("Error in algebraic_control_score:", e)
        return 0.0

def algebraic_threat_score(graph):
    if graph.number_of_nodes() == 0:
        return 0.0
    try:
        A = nx.adjacency_matrix(graph).toarray().astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        lambda_max = np.max(eigenvalues)
        eigen_centrality = np.abs(eigenvectors[:, np.argmax(eigenvalues)]).max()
        return lambda_max + eigen_centrality
    except Exception as e:
        print("Error in algebraic_threat_score:", e)
        return 0.0

def algebraic_strategic_score(graph):
    if graph.number_of_nodes() == 0:
        return 0.0
    try:
        if graph.is_directed():
            graph = graph.to_undirected()
        L_norm = nx.normalized_laplacian_matrix(graph).toarray().astype(float)
        eigenvalues, _ = np.linalg.eigh(L_norm)
        lambda1 = eigenvalues[0]
        centralities = nx.betweenness_centrality(graph)
        max_betweenness = max(centralities.values()) if centralities else 0
        return lambda1 + max_betweenness
    except Exception as e:
        print("Error in algebraic_strategic_score:", e)
        return 0.0

def algebraic_hybrid_score(graphs):
    try:
        for g in graphs:
            if g.number_of_nodes() == 0:
                return 0.0
        undirected_graphs = [g.to_undirected() if g.is_directed() else g for g in graphs]
        tensor_graph = nx.tensor_product(undirected_graphs[0], undirected_graphs[1])
        for g in undirected_graphs[2:]:
            tensor_graph = nx.tensor_product(tensor_graph, g)
        if tensor_graph.number_of_nodes() == 0:
            return 0.0
        A = nx.adjacency_matrix(tensor_graph).toarray().astype(float)
        eigenvalues, _ = np.linalg.eigh(A)
        return np.max(eigenvalues)
    except Exception as e:
        print("Error in algebraic_hybrid_score:", e)
        return 0.0

# ---------------------------
# Algebraic Refinement Functors
# ---------------------------
class AlgebraicControlRefinementFunctor:
    def __call__(self, control_graph):
        return algebraic_control_score(control_graph)
    
class AlgebraicThreatRefinementFunctor:
    def __call__(self, threat_graph):
        return algebraic_threat_score(threat_graph)
    
class AlgebraicStrategicRefinementFunctor:
    def __call__(self, strategic_graph):
        return algebraic_strategic_score(strategic_graph)
    
class AlgebraicHybridRefinementFunctor:
    def __call__(self, graphs):
        return algebraic_hybrid_score(graphs)

# ---------------------------
# Helper Functions
# ---------------------------
def normalize_eval(eval_result):
    try:
        if eval_result['type'] == 'cp':
            return eval_result['value'] / 100.0
        else:
            return 1.0 if eval_result['value'] > 0 else -1.0
    except Exception as e:
        print("Error normalizing evaluation:", e)
        return 0.0

def is_good_position(position, min_pieces=5):
    board = position.board
    if board.is_game_over():
        return False
    piece_count = len(board.piece_map())
    if piece_count < min_pieces:
        return False
    cg = GraphConstructionFunctor("control")(position)
    if cg.number_of_nodes() == 0:
        return False
    return True

def generate_random_position(depth=10):
    board = chess.Board()
    for _ in range(depth):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    return ChessPosition(board.fen())

def generate_random_positions(num_positions=50, depth=10, max_attempts=200):
    positions = []
    attempts = 0
    while len(positions) < num_positions and attempts < max_attempts:
        pos = generate_random_position(depth)
        if is_good_position(pos):
            positions.append(pos)
        attempts += 1
    if len(positions) < num_positions:
        print(f"Warning: Only generated {len(positions)} good positions out of requested {num_positions}.")
    return positions

def compute_similarity_metrics(position1, position2):
    try:
        control_graph1   = control_functor(position1)
        threat_graph1    = threat_functor(position1)
        strategic_graph1 = strategic_functor(position1)
        hybrid_graph1    = hybrid_functor(position1)
        
        control_graph2   = control_functor(position2)
        threat_graph2    = threat_functor(position2)
        strategic_graph2 = strategic_functor(position2)
        hybrid_graph2    = hybrid_functor(position2)
        
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

def visualize_positions(position1, position2):
    svg1 = chess.svg.board(position1.board, size=300)
    svg2 = chess.svg.board(position2.board, size=300)
    return svg1, svg2

# --- New Helper: Normalization using pre-calculated bounds ---
def normalize_metric(value, lower, upper):
    """Apply min-max normalization. If range is 0, return 0.5 as neutral."""
    if upper - lower > 0:
        return (value - lower) / (upper - lower)
    else:
        return 0.5

# ---------------------------
# Experiment, Visualization, and Saving Top Comparisons
# ---------------------------
def run_experiments(num_positions=50, depth=10, num_sample_pairs=200, top_n=5):
    """
    Generate random positions, compute similarity metrics on random pairs,
    normalize these metrics using pre-calculated bounds (from metric_bounds.json),
    compute an aggregate normalized difference, filter out the top-N most similar board pairs,
    and save each top comparison's boards and both raw and normalized metrics into separate folders.
    """
    # Base output directory (adjust if necessary)
    base_output_dir = "/Users/ashutoshganguly/Desktop/Research Papers/abid_ashutosh_papers/chess_board_distance/deepseekr1/"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load pre-calculated bounds
    bounds_path = os.path.join(base_output_dir, "metric_bounds.json")
    with open(bounds_path, "r") as f:
        metric_bounds = json.load(f)
    
    positions = generate_random_positions(num_positions, depth)
    if not positions:
        print("No valid positions generated.")
        return
    
    raw_metrics_all = {
        "control_diff":   [],
        "threat_diff":    [],
        "strategic_diff": [],
        "hybrid_diff":    [],
        "eval_diff":      []
    }
    comparisons = []  # List to store (pos1, pos2, raw_metrics, normalized_metrics, aggregate_norm) for each valid pair
    total_positions = len(positions)
    
    # Sample pairs and compute raw metrics
    for _ in range(num_sample_pairs):
        i, j = random.sample(range(total_positions), 2)
        metrics_diff = compute_similarity_metrics(positions[i], positions[j])
        if metrics_diff is not None:
            for key in raw_metrics_all:
                raw_metrics_all[key].append(metrics_diff[key])
            comparisons.append({
                "pos1": positions[i],
                "pos2": positions[j],
                "raw_metrics": metrics_diff
            })
    
    if not comparisons:
        print("No valid position pairs for similarity metrics computation.")
        return

    # For each comparison, normalize the metrics using the pre-calculated bounds and compute an aggregate normalized difference.
    METRIC_KEYS = ["control_diff", "threat_diff", "strategic_diff", "hybrid_diff", "eval_diff"]
    for comp in comparisons:
        norm_metrics = {}
        for key in METRIC_KEYS:
            lower = metric_bounds[key]["min"]
            upper = metric_bounds[key]["max"]
            norm_metrics[key] = normalize_metric(comp["raw_metrics"][key], lower, upper)
        comp["normalized_metrics"] = norm_metrics
        comp["aggregate_norm"] = sum(norm_metrics.values())
    
    # Sort comparisons by aggregate_norm ascending (lower means more similar) and select top_n comparisons
    top_comparisons = sorted(comparisons, key=lambda c: c["aggregate_norm"])[:top_n]
    
    print("Top Similarity Comparisons (using normalized metrics):")
    for idx, comp in enumerate(top_comparisons):
        print(f"Pair {idx+1}: Aggregate Normalized Difference = {comp['aggregate_norm']:.4f}")
        for key in METRIC_KEYS:
            print(f"   {key}: raw = {comp['raw_metrics'][key]:.4f}, normalized = {comp['normalized_metrics'][key]:.4f}")
        print()
    
    # Plot histograms for all computed raw metric differences
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for ax, key in zip(axs, METRIC_KEYS):
        ax.hist(raw_metrics_all[key], bins=15, color='skyblue', edgecolor='black')
        ax.set_title(key)
        ax.set_xlabel("Difference")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    # Save each top comparison into its own folder
    for idx, comp in enumerate(top_comparisons):
        folder_name = os.path.join(base_output_dir, f"top_comparison_{idx+1}")
        os.makedirs(folder_name, exist_ok=True)
        print(f"Creating folder: {folder_name}")
        
        # Save boards as SVG files
        svg1, svg2 = visualize_positions(comp["pos1"], comp["pos2"])
        board1_path = os.path.join(folder_name, "board1.svg")
        board2_path = os.path.join(folder_name, "board2.svg")
        with open(board1_path, "w") as f:
            f.write(svg1)
        with open(board2_path, "w") as f:
            f.write(svg2)
        print(f"Saved board SVGs to {board1_path} and {board2_path}")
        
        # Save FEN strings
        fen_path = os.path.join(folder_name, "boards.txt")
        with open(fen_path, "w") as f:
            f.write("Position 1 FEN:\n" + comp["pos1"].board.fen() + "\n\n")
            f.write("Position 2 FEN:\n" + comp["pos2"].board.fen() + "\n")
        print(f"Saved FEN strings to {fen_path}")
        
        # Save both raw and normalized metrics (including aggregate_norm) as JSON
        metrics_to_save = {
            "raw_metrics": comp["raw_metrics"],
            "normalized_metrics": comp["normalized_metrics"],
            "aggregate_normalized_diff": comp["aggregate_norm"]
        }
        metrics_path = os.path.join(folder_name, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"Saved metrics to {metrics_path}")
    
    # Visualize one pair of top comparisons for interactive inspection
    sample = random.choice(top_comparisons)
    print("\nVisualizing one top similar pair of boards:")
    svg1, svg2 = visualize_positions(sample["pos1"], sample["pos2"])
    html_str = f"""
    <div style="display: flex; gap: 20px;">
        <div>{svg1}</div>
        <div>{svg2}</div>
    </div>
    """
    display(SVG(html_str))

# ---------------------------
# Initialization and Execution
# ---------------------------
# Update the path to your Stockfish executable as needed (for macOS)
stockfish_path = "/opt/homebrew/bin/stockfish"
evaluator = EvaluationFunctor(stockfish_path)

# Define graph functors for each aspect
control_functor    = GraphConstructionFunctor("control")
threat_functor     = GraphConstructionFunctor("threat")
strategic_functor  = GraphConstructionFunctor("strategic")
hybrid_functor     = GraphConstructionFunctor("hybrid")

# Define algebraic refinement functors
control_refinement   = AlgebraicControlRefinementFunctor()
threat_refinement    = AlgebraicThreatRefinementFunctor()
strategic_refinement = AlgebraicStrategicRefinementFunctor()
hybrid_refinement    = AlgebraicHybridRefinementFunctor()

if __name__ == "__main__":
    run_experiments(num_positions=150, depth=25, num_sample_pairs=50, top_n=10)
