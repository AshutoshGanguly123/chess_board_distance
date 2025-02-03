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
import os
import json
import matplotlib.pyplot as plt

# ---------------------------
# Chess Position and Morphism
# ---------------------------
class ChessPosition:
    def __init__(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen)
    
    def morphism(self, move):
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
        G = nx.Graph()  # using undirected for strategic measures
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
        threat_graph = self._build_threat_graph(position).to_undirected()  # for consistency
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
        # Compute graph-based metrics
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
        
        # Graph-based similarity (sim score) based solely on the four metrics:
        control_diff   = abs(control_score1 - control_score2)
        threat_diff    = abs(threat_score1 - threat_score2)
        strategic_diff = abs(strategic_score1 - strategic_score2)
        hybrid_diff    = abs(hybrid_score1 - hybrid_score2)
        graph_diff = control_diff + threat_diff + strategic_diff + hybrid_diff
        
        # Engine evaluation differences computed separately
        eval1 = normalize_eval(evaluator(position1))
        eval2 = normalize_eval(evaluator(position2))
        # Discard pair if evals disagree on winning side
        if eval1 * eval2 < 0:
            return None
        eval_diff = abs(eval1 - eval2)
        
        return {
            "control_diff": control_diff,
            "threat_diff": threat_diff,
            "strategic_diff": strategic_diff,
            "hybrid_diff": hybrid_diff,
            "graph_diff": graph_diff,  # Graph-based similarity score
            "eval_diff": eval_diff,    # Engine evaluation difference
            "eval1": eval1,
            "eval2": eval2
        }
    except Exception as e:
        print("Error computing similarity metrics for pair:", e)
        return None

def visualize_positions(position1, position2):
    # Increase size parameter to produce larger SVG boards
    svg1 = chess.svg.board(position1.board, size=800)
    svg2 = chess.svg.board(position2.board, size=800)
    return svg1, svg2

# --- New Helper: Normalization using pre-calculated bounds ---
def normalize_metric(value, lower, upper):
    if upper - lower > 0:
        return (value - lower) / (upper - lower)
    else:
        return 0.5

# ---------------------------
# Experiment, Visualization, and Saving Sorted Comparisons
# ---------------------------
def run_experiments(num_positions=50, depth=10, num_sample_pairs=200, top_n=10):
    """
    Generate random positions, compute similarity metrics on random pairs,
    normalize the graph-based metrics using pre-calculated bounds (from metric_bounds.json),
    compute an aggregate normalized graph difference (excluding engine eval),
    combine it with the engine eval difference to compute a final combined score,
    and then filter out pairs where the engine evals do not agree on the winning side.
    Finally, sort all comparisons by the combined score and save each comparison's boards and metrics
    into separate folders.
    Additionally, plot a regression graph (scatter plot with a regression line)
    comparing the aggregate normalized graph difference and the engine eval difference.
    """
    base_output_dir = "/Users/ashutoshganguly/Desktop/Research Papers/abid_ashutosh_papers/chess_board_distance/deepseekr1/"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load pre-calculated bounds for graph-based metrics
    bounds_path = os.path.join(base_output_dir, "metric_bounds.json")
    with open(bounds_path, "r") as f:
        metric_bounds = json.load(f)
    
    positions = generate_random_positions(num_positions, depth)
    if not positions:
        print("No valid positions generated.")
        return
    
    comparisons = []  # List to store each valid comparison with raw and normalized metrics.
    total_positions = len(positions)
    
    for _ in range(num_sample_pairs):
        i, j = random.sample(range(total_positions), 2)
        metrics_diff = compute_similarity_metrics(positions[i], positions[j])
        if metrics_diff is not None:
            comparisons.append({
                "pos1": positions[i],
                "pos2": positions[j],
                "raw_metrics": metrics_diff
            })
    
    if not comparisons:
        print("No valid comparisons computed.")
        return

    # Define keys for graph-based metrics (excluding engine eval)
    GRAPH_KEYS = ["control_diff", "threat_diff", "strategic_diff", "hybrid_diff"]
    
    # Normalize the graph-based metrics and compute an aggregate normalized graph difference
    for comp in comparisons:
        norm_metrics = {}
        for key in GRAPH_KEYS:
            lower = metric_bounds[key]["min"]
            upper = metric_bounds[key]["max"]
            norm_metrics[key] = normalize_metric(comp["raw_metrics"][key], lower, upper)
        comp["normalized_graph_metrics"] = norm_metrics
        comp["aggregate_norm_graph"] = sum(norm_metrics.values())
        # Engine eval difference is already computed (and stored in raw_metrics)
        comp["eval_diff"] = comp["raw_metrics"]["eval_diff"]
    
    # Compute a combined score: here we simply add the aggregate normalized graph score and the engine eval diff.
    for comp in comparisons:
        comp["combined_score"] = comp["aggregate_norm_graph"] + comp["eval_diff"]
    
    # Sort comparisons by the combined score (lowest means most similar in both measures)
    sorted_comparisons = sorted(comparisons, key=lambda c: c["combined_score"])
    
    # Select the top_n comparisons
    top_comparisons = sorted_comparisons[:top_n]
    
    print("Top Similarity Comparisons (Combined Graph and Engine Eval):")
    for idx, comp in enumerate(top_comparisons):
        print(f"Pair {idx+1}: Combined Score = {comp['combined_score']:.4f}")
        print(f"   Aggregate Graph Norm Diff = {comp['aggregate_norm_graph']:.4f}, Engine Eval Diff = {comp['eval_diff']:.4f}")
        for key in GRAPH_KEYS:
            print(f"      {key}: raw = {comp['raw_metrics'][key]:.4f}, normalized = {comp['normalized_graph_metrics'][key]:.4f}")
        print(f"   Engine Evals: Board1 = {comp['raw_metrics']['eval1']:.4f}, Board2 = {comp['raw_metrics']['eval2']:.4f}")
        print()
    
    # Save each top comparison into its own folder
    for idx, comp in enumerate(top_comparisons):
        folder_name = os.path.join(base_output_dir, f"top_comparison_{idx+1}")
        os.makedirs(folder_name, exist_ok=True)
        print(f"Storing comparison #{idx+1} in folder: {folder_name}")
        
        # Save boards as SVG files (full screen size)
        svg1, svg2 = visualize_positions(comp["pos1"], comp["pos2"])
        with open(os.path.join(folder_name, "board1.svg"), "w") as f:
            f.write(svg1)
        with open(os.path.join(folder_name, "board2.svg"), "w") as f:
            f.write(svg2)
        
        # Save FEN strings
        with open(os.path.join(folder_name, "boards.txt"), "w") as f:
            f.write("Position 1 FEN:\n" + comp["pos1"].board.fen() + "\n\n")
            f.write("Position 2 FEN:\n" + comp["pos2"].board.fen() + "\n")
        
        # Save both raw and normalized metrics (including combined score) as JSON
        metrics_to_save = {
            "raw_metrics": comp["raw_metrics"],
            "normalized_graph_metrics": comp["normalized_graph_metrics"],
            "aggregate_normalized_graph_diff": comp["aggregate_norm_graph"],
            "engine_eval_diff": comp["eval_diff"],
            "combined_score": comp["combined_score"]
        }
        with open(os.path.join(folder_name, "metrics.json"), "w") as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"Stored comparison in folder: {folder_name}")
    
    # ---- Regression Plot: Plot all comparisons (graph vs. engine eval differences) ----
    # For each comparison, get the aggregate normalized graph diff and the engine eval diff.
    x_vals = np.array([comp["aggregate_norm_graph"] for comp in comparisons])
    y_vals = np.array([comp["eval_diff"] for comp in comparisons])
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_vals, y_vals, c=y_vals, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.xlabel("Aggregate Normalized Graph Diff")
    plt.ylabel("Engine Eval Diff")
    plt.title("Regression Plot: Graph-based Similarity vs. Engine Evaluation Difference")
    plt.colorbar(scatter, label="Engine Eval Diff")
    
    # Compute linear regression
    if len(x_vals) > 1:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        poly_eqn = np.poly1d(coeffs)
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        plt.plot(x_line, poly_eqn(x_line), color='red', linewidth=2, label=f"Regression Line\ny={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "regression_plot.png"))
    plt.show()
    print(f"Saved regression plot to {os.path.join(base_output_dir, 'regression_plot.png')}")
    
    print(f"Stored {len(sorted_comparisons)} comparisons sorted from most similar to least similar based on combined score.")

# ---------------------------
# Initialization and Execution
# ---------------------------
stockfish_path = "/opt/homebrew/bin/stockfish"  # Adjust if needed
evaluator = EvaluationFunctor(stockfish_path)

control_functor    = GraphConstructionFunctor("control")
threat_functor     = GraphConstructionFunctor("threat")
strategic_functor  = GraphConstructionFunctor("strategic")
hybrid_functor     = GraphConstructionFunctor("hybrid")

control_refinement   = AlgebraicControlRefinementFunctor()
threat_refinement    = AlgebraicThreatRefinementFunctor()
strategic_refinement = AlgebraicStrategicRefinementFunctor()
hybrid_refinement    = AlgebraicHybridRefinementFunctor()

if __name__ == "__main__":
    run_experiments(num_positions=150, depth=25, num_sample_pairs=100, top_n=10)
