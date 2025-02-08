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

def generate_random_positions(num_positions,depth):
    positions = []
    attempts = 0
    max_attempts = num_positions
    while len(positions) < num_positions and attempts < max_attempts:
        pos = generate_random_position(depth)
        if is_good_position(pos):
            positions.append(pos)
        attempts += 1
    if len(positions) < num_positions:
        print(f"Warning: Only generated {len(positions)} good positions out of requested {num_positions}.")
    return positions

def compute_board_metric(position):
    # Compute individual board's graph-based metric (aggregate of scores)
    control_score = control_refinement(control_functor(position))
    threat_score = threat_refinement(threat_functor(position))
    strategic_score = strategic_refinement(strategic_functor(position))
    hybrid_score = hybrid_refinement([control_functor(position), threat_functor(position), strategic_functor(position)])
    return control_score + threat_score + strategic_score + hybrid_score

def compute_similarity_metrics(position1, position2):
    try:
        # Compute graph-based metrics for each board pair
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
        
        # Graph-based similarity (our sim score) based solely on the four metrics:
        control_diff   = abs(control_score1 - control_score2)
        threat_diff    = abs(threat_score1 - threat_score2)
        strategic_diff = abs(strategic_score1 - strategic_score2)
        hybrid_diff    = abs(hybrid_score1 - hybrid_score2)
        graph_diff = control_diff + threat_diff + strategic_diff + hybrid_diff
        
        # Compute engine evaluations separately
        eval1 = normalize_eval(evaluator(position1))
        eval2 = normalize_eval(evaluator(position2))
        # Discard pair if evaluations disagree on the winning side
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
    # Increase board size to 800 for full-screen view
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
# Plot Regression for Filtered Boards
# ---------------------------
def plot_filtered_board_metrics(num_positions=500, depth=25, top_n_boards=100):
    """
    Generate a set of boards, compute for each the engine eval and graph metric,
    then sort boards by the absolute difference between these two.
    Select the top_n_boards with the smallest difference and plot:
      - x-axis: Board index (1 to top_n_boards)
      - y-axis: Two sets of points: one for engine eval and one for graph metric.
    Additionally, compute and print regression metrics (slope, intercept, RMSE, R²)
    between the graph metrics and engine evals.
    """
    positions = generate_random_positions(num_positions, depth)
    if not positions:
        print("No valid positions generated for board metrics plot.")
        return
    board_data = []
    for pos in positions:
        eval_val = normalize_eval(evaluator(pos))
        graph_val = compute_board_metric(pos)
        diff = abs(eval_val - graph_val)
        board_data.append((pos, eval_val, graph_val, diff))
    
    # Sort boards by the absolute difference (lower means engine eval and graph metric match closely)
    board_data_sorted = sorted(board_data, key=lambda x: x[3])
    selected_data = board_data_sorted[:top_n_boards]
    
    board_indices = np.arange(1, len(selected_data)+1)
    engine_evals = np.array([data[1] for data in selected_data])
    graph_metrics = np.array([data[2] for data in selected_data])
    
    plt.figure(figsize=(12, 6))
    plt.scatter(board_indices, engine_evals, color='blue', label='Engine Eval', s=60, marker='^')
    plt.scatter(board_indices, graph_metrics, color='red', label='Graph Metric', s=60, marker='o')
    plt.xlabel("Board Index (Filtered)")
    plt.ylabel("Score")
    plt.title("Filtered Board Metrics: Engine Eval vs. Graph Metric")
    plt.legend()
    
    # Perform linear regression: regress engine_evals against graph_metrics
    coeffs = np.polyfit(graph_metrics, engine_evals, 1)
    poly_eqn = np.poly1d(coeffs)
    x_line = np.linspace(min(graph_metrics), max(graph_metrics), 100)
    y_line = poly_eqn(x_line)
    
    y_pred = poly_eqn(graph_metrics)
    residuals = engine_evals - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum((engine_evals - y_pred)**2)
    ss_tot = np.sum((engine_evals - np.mean(engine_evals))**2)
    r_squared = 1 - ss_res/ss_tot if ss_tot != 0 else 0
    
    print("Regression Metrics for Filtered Boards:")
    print(f"  Slope: {coeffs[0]:.4f}")
    print(f"  Intercept: {coeffs[1]:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r_squared:.4f}")
    
    plt.plot(x_line, y_line, color='green', linewidth=2, 
             label=f"Regression Line\ny={coeffs[0]:.2f}x+{coeffs[1]:.2f}\nRMSE={rmse:.2f}, R²={r_squared:.2f}")
    plt.legend()
    
    plot_path = os.path.join("/Users/ashutoshganguly/Desktop/Research Papers/abid_ashutosh_papers/chess_board_distance/deepseekr1/", "filtered_board_metrics_regression.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved filtered board metrics regression plot to {plot_path}")

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
    # You can still run your experiment functions if desired.
    # For example, to run experiments and store top comparisons:
    # run_experiments(num_positions=150, depth=25, num_sample_pairs=300, top_n=10)
    
    # Now, plot regression for the best 100 boards (filtered by closest match between engine eval and graph metric)
    plot_filtered_board_metrics(num_positions=1000, depth=20, top_n_boards=500)
