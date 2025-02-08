import networkx as nx
import numpy as np
import chess

def algebraic_control_score(graph):
    """
    Computes control features from the graph:
      - "fiedler": the second-smallest eigenvalue of the Laplacian
      - "fiedler_centrality": the maximum absolute value of the Fiedler vector
    Returns these as a dictionary.
    """
    if graph.number_of_nodes() == 0:
        return {"fiedler": 0.0, "fiedler_centrality": 0.0}
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
        return {"fiedler": lambda2, "fiedler_centrality": centrality}
    except Exception as e:
        print("Error in algebraic_control_score:", e)
        return {"fiedler": 0.0, "fiedler_centrality": 0.0}

def algebraic_threat_score(graph):
    """
    Computes threat features from the graph:
      - "spectral_radius": the maximum eigenvalue of the adjacency matrix
      - "threat_centrality": the maximum absolute value of the corresponding eigenvector
    Returns these as a dictionary.
    """
    if graph.number_of_nodes() == 0:
        return {"spectral_radius": 0.0, "threat_centrality": 0.0}
    try:
        A = nx.adjacency_matrix(graph).toarray().astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        lambda_max = np.max(eigenvalues)
        eigen_centrality = np.abs(eigenvectors[:, np.argmax(eigenvalues)]).max()
        return {"spectral_radius": lambda_max, "threat_centrality": eigen_centrality}
    except Exception as e:
        print("Error in algebraic_threat_score:", e)
        return {"spectral_radius": 0.0, "threat_centrality": 0.0}

# --- Enhanced Strategic Invariants for a Side-Specific Graph ---

def central_dominance_score(strategic_graph):
    """
    Computes the average of 1/(1+distance) from board center (3.5,3.5).
    A higher value indicates that the side’s pieces tend to be closer to the center.
    """
    if strategic_graph.number_of_nodes() == 0:
        print('Strategic graph empty in central dominance')
        return 0.0
    total = 0.0
    count = 0
    for node in strategic_graph.nodes():
        file = chess.square_file(node)
        rank = chess.square_rank(node)
        dist = ((file - 3.5)**2 + (rank - 3.5)**2)**0.5
        total += 1.0 / (1 + dist)
        count += 1
    return total / count

def king_shelter_integrity(position, strategic_graph, side):
    """
    For a side-specific graph, finds the king (for the given side) and computes the density
    of its 3x3 neighborhood (Chebyshev distance ≤1). Higher density means a more solid shelter.
    """
    board = position.board
    king_sq = board.king(side)
    if king_sq is None or king_sq not in strategic_graph:
        print('King not found in strategic graph for side:', side)
        return 0.0
    neighbors = [n for n in strategic_graph.nodes() if max(
        abs(chess.square_file(n) - chess.square_file(king_sq)),
        abs(chess.square_rank(n) - chess.square_rank(king_sq))
    ) <= 1]
    subG = strategic_graph.subgraph(neighbors)
    n = subG.number_of_nodes()
    if n <= 1:
        return 0.0
    return nx.density(subG)

def piece_coordination_index(strategic_graph):
    """
    Returns the average clustering coefficient of the strategic graph.
    """
    try:
        return nx.average_clustering(strategic_graph)
    except Exception as e:
        print("Error computing piece_coordination_index:", e)
        return 0.0

def mobility_synergy_score(strategic_graph):
    """
    Returns the overall density of the strategic graph as a proxy for synergy.
    """
    try:
        return nx.density(strategic_graph)
    except Exception as e:
        print("Error computing mobility_synergy_score:", e)
        return 0.0

def strategic_imbalance_metric(strategic_graph):
    """
    Computes the ratio of maximum node degree to average node degree.
    """
    degrees = [d for n, d in strategic_graph.degree()]
    if not degrees:
        return 0.0
    avg_deg = sum(degrees) / len(degrees)
    max_deg = max(degrees)
    if avg_deg == 0:
        return 0.0
    return max_deg / avg_deg

def enhanced_strategic_score(position, strategic_graph, side):
    """
    Returns a dictionary containing five strategic invariants:
      - "central_dominance"
      - "king_shelter"
      - "coordination"
      - "mobility_synergy"
      - "imbalance"
    """
    cds = central_dominance_score(strategic_graph)
    ksi = king_shelter_integrity(position, strategic_graph, side)
    pci = piece_coordination_index(strategic_graph)
    mss = mobility_synergy_score(strategic_graph)
    sim = strategic_imbalance_metric(strategic_graph)
    return {
        "central_dominance": cds,
        "king_shelter": ksi,
        "coordination": pci,
        "mobility_synergy": mss,
        "imbalance": sim
    }

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

# --- Algebraic Refinement Functors ---

class AlgebraicControlRefinementFunctor:
    def __call__(self, control_graph):
        return algebraic_control_score(control_graph)
    
class AlgebraicThreatRefinementFunctor:
    def __call__(self, threat_graph):
        return algebraic_threat_score(threat_graph)
    
class AlgebraicStrategicRefinementFunctor:
    def __call__(self, strategic_graph, position=None, side=None):
        if position is None or side is None:
            return {}
        return enhanced_strategic_score(position, strategic_graph, side)
    
class AlgebraicHybridRefinementFunctor:
    def __call__(self, graphs):
        return algebraic_hybrid_score(graphs)
