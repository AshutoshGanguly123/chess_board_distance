#!/usr/bin/env python
"""
Enhanced Positional Graph Implementation for Deep Chess Analysis

Features:
 - Pawn Structure: Each pawn is annotated with roles (isolated, backward, passed, chain member).
 - Zone Assignment: Each board square is assigned a zone ("center", "kingside", "queenside")
   based on a chess–theoretic function.
 - Additional Invariants: Composite invariants are computed including:
     * Pawn island count
     * Passed pawn score
     * Space score
     * Shield index
     * Attackers' proximity
     * Spectral invariants computed on the overall graph (e.g., Fiedler value, eigenvector centrality, clustering)
     * Influence subgraph invariants (built from legal moves using uniform weights) such as:
         - Influence Fiedler Value
         - Global Efficiency
         - Average Degree
         - Average Closeness
         - Average Betweenness
         - Average Clustering
 - Legal Move Enforcement: Influence edges are built using only legal moves.
 - Standardized Influence: Instead of piece-specific distances, all influence edges now use a uniform weight (1).
 - Material weights are removed so that the graph reflects pure positional connectivity.
 - A multilayered approach distinguishes latent (static) and dynamic (legal-move) connectivity.
    
It uses python‑chess for board representation, NetworkX for graph construction, and matplotlib for visualization.
"""

import chess
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Piece-Specific Distance Measure (unused in influence edges now)
# ------------------------------------------------------------------------------
def piece_distance(piece_symbol, from_square, to_square):
    """
    Compute a distance measure between two squares, specific to the piece type.
    (This function is retained for other parts of the analysis but is not used for influence edges.)
    
    Rules:
      - Knight ('N'): Euclidean distance.
      - Bishop ('B'): Chebyshev distance (max(|dx|, |dy|)).
      - Rook ('R'): Manhattan distance.
      - Queen ('Q'): If diagonal (dx == dy) use Chebyshev; otherwise, Manhattan.
      - King ('K'): 1 (always moves one square).
      - Pawn ('P'): Manhattan distance.
    """
    coord1 = square_to_coord(from_square)
    coord2 = square_to_coord(to_square)
    dx = abs(coord1[0] - coord2[0])
    dy = abs(coord1[1] - coord2[1])
    piece_type = piece_symbol.upper()
    
    if piece_type == 'N':
        return np.sqrt(dx**2 + dy**2)
    elif piece_type == 'B':
        return max(dx, dy)
    elif piece_type == 'R':
        return dx + dy
    elif piece_type == 'Q':
        if dx == dy:
            return max(dx, dy)
        else:
            return dx + dy
    elif piece_type == 'K':
        return 1
    elif piece_type == 'P':
        return dx + dy
    else:
        return dx + dy

# ------------------------------------------------------------------------------
# Helper Functions for Geometry and Control
# ------------------------------------------------------------------------------
def square_to_coord(square_name):
    file = ord(square_name[0]) - ord('a')
    rank = int(square_name[1]) - 1
    return (file, rank)

def manhattan_distance(coord1, coord2):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def control_value(square1, square2, board):
    """
    Return 1 if both squares (given by their names) are occupied on the provided board;
    otherwise return 0.
    """
    if board.piece_at(chess.parse_square(square1)) is not None and \
       board.piece_at(chess.parse_square(square2)) is not None:
        return 1
    else:
        return 0

def square_weight(square):
    central_squares = {"d4", "d5", "e4", "e5"}
    if square in central_squares:
        return 2.0
    file, rank = square_to_coord(square)
    if file == 0 or file == 7 or rank == 0 or rank == 7:
        return 0.5
    return 1.0

def control_factor(board, square, color):
    """
    Count the number of attackers on the given square from the specified color.
    """
    attackers = board.attackers(color, chess.parse_square(square))
    return len(attackers)

# ------------------------------------------------------------------------------
# Pawn Role Determination Using Chess Theory
# ------------------------------------------------------------------------------
def get_pawn_roles(board, square):
    roles = []
    sq_index = chess.parse_square(square)
    piece = board.piece_at(sq_index)
    if piece is None or piece.symbol().lower() != 'p':
        return roles
    color = piece.color
    file_index = chess.square_file(sq_index)
    rank_index = chess.square_rank(sq_index)
    
    adjacent_files = []
    if file_index > 0:
        adjacent_files.append(file_index - 1)
    if file_index < 7:
        adjacent_files.append(file_index + 1)
    
    # Isolated Pawn:
    isolated = True
    for other_sq in chess.SQUARES:
        if other_sq == sq_index:
            continue
        other_piece = board.piece_at(other_sq)
        if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color == color:
            other_file = chess.square_file(other_sq)
            if other_file in adjacent_files:
                isolated = False
                break
    if isolated:
        roles.append("isolated")
    
    # Passed Pawn:
    passed = True
    if color == chess.WHITE:
        for other_sq in chess.SQUARES:
            other_piece = board.piece_at(other_sq)
            if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color != color:
                other_file = chess.square_file(other_sq)
                other_rank = chess.square_rank(other_sq)
                if other_file in ([file_index] + adjacent_files) and other_rank > rank_index:
                    passed = False
                    break
    else:
        for other_sq in chess.SQUARES:
            other_piece = board.piece_at(other_sq)
            if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color != color:
                other_file = chess.square_file(other_sq)
                other_rank = chess.square_rank(other_sq)
                if other_file in ([file_index] + adjacent_files) and other_rank < rank_index:
                    passed = False
                    break
    if passed:
        roles.append("passed")
    
    # Chain Member:
    chain_member = False
    if color == chess.WHITE:
        if rank_index > 0:
            if file_index > 0:
                sq_support = chess.square(file_index - 1, rank_index - 1)
                other_piece = board.piece_at(sq_support)
                if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color == color:
                    chain_member = True
            if file_index < 7:
                sq_support = chess.square(file_index + 1, rank_index - 1)
                other_piece = board.piece_at(sq_support)
                if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color == color:
                    chain_member = True
    else:
        if rank_index < 7:
            if file_index > 0:
                sq_support = chess.square(file_index - 1, rank_index + 1)
                other_piece = board.piece_at(sq_support)
                if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color == color:
                    chain_member = True
            if file_index < 7:
                sq_support = chess.square(file_index + 1, rank_index + 1)
                other_piece = board.piece_at(sq_support)
                if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color == color:
                    chain_member = True
    if chain_member:
        roles.append("chain_member")
    
    # Backward Pawn:
    backward = False
    if color == chess.WHITE:
        support_found = False
        for other_sq in chess.SQUARES:
            if other_sq == sq_index:
                continue
            other_piece = board.piece_at(other_sq)
            if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color == color:
                other_file = chess.square_file(other_sq)
                other_rank = chess.square_rank(other_sq)
                if other_file in adjacent_files and other_rank >= rank_index:
                    support_found = True
                    break
        if not support_found:
            backward = True
    else:
        support_found = False
        for other_sq in chess.SQUARES:
            if other_sq == sq_index:
                continue
            other_piece = board.piece_at(other_sq)
            if other_piece is not None and other_piece.symbol().lower() == 'p' and other_piece.color == color:
                other_file = chess.square_file(other_sq)
                other_rank = chess.square_rank(other_sq)
                if other_file in adjacent_files and other_rank <= rank_index:
                    support_found = True
                    break
        if not support_found:
            backward = True
    if backward:
        roles.append("backward")
    
    return roles

# ------------------------------------------------------------------------------
# Zone Assignment Using Chess Theory
# ------------------------------------------------------------------------------
def get_zone(square):
    file_index, rank_index = square_to_coord(square)
    if file_index in [3, 4] and rank_index in [3, 4]:
        return "center"
    elif file_index >= 4:
        return "kingside"
    else:
        return "queenside"

# ------------------------------------------------------------------------------
# Main Class: PositionalGraph
# ------------------------------------------------------------------------------
class PositionalGraph:
    def __init__(self, board, zones=None):
        """
        Initialize the positional graph using a python‑chess Board instance.
        Optionally, a dictionary mapping zone names to sets of square names may be provided.
        If not, zones are assigned using get_zone().
        """
        self.board = board
        self.graph = nx.Graph()
        self.board_squares = [chess.square_name(sq) for sq in chess.SQUARES]
        if zones is None:
            self.zones = {"center": set(), "kingside": set(), "queenside": set()}
            for square in self.board_squares:
                zone_label = get_zone(square)
                if zone_label not in self.zones:
                    self.zones[zone_label] = set()
                self.zones[zone_label].add(square)
        else:
            self.zones = zones
        self._build_graph()

    def _build_graph(self):
        """Construct the complete positional graph with nodes and all edge types."""
        self._add_board_nodes()
        self._add_pawn_nodes()
        self._add_zone_nodes()
        self._add_adjacency_edges()
        self._add_pawn_support_edges()
        self._add_influence_edges()  # Build influence edges using the current board state.
        self._add_zone_edges()

    def _add_board_nodes(self):
        for square in self.board_squares:
            self.graph.add_node(square, type="square")

    def _add_pawn_nodes(self):
        for square in self.board_squares:
            sq = chess.parse_square(square)
            piece = self.board.piece_at(sq)
            if piece is not None and piece.symbol().lower() == 'p':
                roles = get_pawn_roles(self.board, square)
                pawn_node = f"pawn_{square}"
                self.graph.add_node(pawn_node, type="pawn", roles=roles, square=square)

    def _add_zone_nodes(self):
        for zone_name in self.zones.keys():
            self.graph.add_node(zone_name, type="zone")

    def _add_adjacency_edges(self):
        for i, s1 in enumerate(self.board_squares):
            for s2 in self.board_squares[i+1:]:
                coord1 = square_to_coord(s1)
                coord2 = square_to_coord(s2)
                if manhattan_distance(coord1, coord2) == 1:
                    cv = control_value(s1, s2, self.board)
                    if cv == 1:
                        weight = cv * (1 + manhattan_distance(coord1, coord2))
                        self.graph.add_edge(s1, s2, type="adjacency", weight=weight)

    def _add_pawn_support_edges(self):
        for square in self.board_squares:
            sq = chess.parse_square(square)
            piece = self.board.piece_at(sq)
            if piece is not None and piece.symbol().lower() == 'p':
                file, rank = square_to_coord(square)
                if piece.color == chess.WHITE:
                    support_coords = [(file - 1, rank + 1), (file + 1, rank + 1)]
                else:
                    support_coords = [(file - 1, rank - 1), (file + 1, rank - 1)]
                for f, r in support_coords:
                    if 0 <= f < 8 and 0 <= r < 8:
                        support_square = chr(ord('a') + f) + str(r + 1)
                        pawn_node = f"pawn_{square}"
                        # Use the old pawn distance here if desired (unchanged)
                        distance = piece_distance('P', square, support_square)
                        w = control_value(square, support_square, self.board)
                        if w == 1:
                            weight = w * (1 + distance)
                            self.graph.add_edge(pawn_node, support_square, type="pawn_support", weight=weight)

    def _add_influence_edges(self):
        """
        For each piece on the board, add influence edges from the piece's square to each square it legally moves to.
        This version uses a standardized approach: each legal move adds an edge with a uniform weight of 1.
        """
        legal_moves = list(self.board.generate_legal_moves())
        moves_from = {}
        for move in legal_moves:
            moves_from.setdefault(move.from_square, []).append(move)
        for square in self.board_squares:
            sq = chess.parse_square(square)
            piece = self.board.piece_at(sq)
            if piece is not None:
                moves = moves_from.get(sq, [])
                for move in moves:
                    target = chess.square_name(move.to_square)
                    # Standardized: each legal move adds an edge with weight = 1 (ignoring piece-specific distances)
                    weight = 1
                    self.graph.add_edge(square, target, type="influence", weight=weight)

    def _add_zone_edges(self):
        """
        Add an edge between each square and its corresponding zone node,
        but only if the square is occupied by a piece.
        """
        for square in self.board_squares:
            if self.board.piece_at(chess.parse_square(square)) is not None:
                zone_label = get_zone(square)
                self.graph.add_edge(square, zone_label, type="zone", weight=1)

    # ------------------------------------------------------------------------------
    # Influence Subgraph Computation by Color
    # ------------------------------------------------------------------------------
    def compute_influence_subgraph_by_color(self, color):
        """
        Compute the influence subgraph for a given color.
        Create a fresh board copy, set its turn, generate legal moves,
        and build a subgraph with only influence edges based on those moves.
        This version also uses a uniform weight (1) for each legal move.
        """
        board_copy = self.board.copy(stack=False)
        board_copy.turn = color
        legal_moves = list(board_copy.generate_legal_moves())
        moves_from = {}
        for move in legal_moves:
            moves_from.setdefault(move.from_square, []).append(move)
        G_inf = nx.Graph()
        G_inf.add_nodes_from(self.graph.nodes(data=True))
        for square in self.board_squares:
            sq = chess.parse_square(square)
            piece = board_copy.piece_at(sq)
            if piece is not None:
                moves = moves_from.get(sq, [])
                for move in moves:
                    target = chess.square_name(move.to_square)
                    weight = 1  # Standardized uniform weight
                    G_inf.add_edge(square, target, type="influence", weight=weight)
        return G_inf

    # ------------------------------------------------------------------------------
    # Invariant Computation Functions
    # ------------------------------------------------------------------------------
    def compute_pawn_island_count(self):
        pawn_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "pawn"]
        pawn_subgraph = self.graph.subgraph(pawn_nodes)
        islands = list(nx.connected_components(pawn_subgraph))
        return len(islands)

    def compute_passed_pawn_score(self):
        score = 0.0
        for node, attr in self.graph.nodes(data=True):
            if attr.get("type") == "pawn":
                square = attr.get("square")
                file, rank = square_to_coord(square)
                if rank >= 4:
                    pawn = self.board.piece_at(chess.parse_square(square))
                    if pawn and pawn.color == chess.WHITE:
                        distance_to_promotion = 7 - rank
                    else:
                        distance_to_promotion = rank
                    score += (8 - distance_to_promotion)
        return score

    def compute_space_score(self, color=chess.WHITE):
        central_squares = self.zones.get("center", set())
        score = 0.0
        for square in central_squares:
            w = square_weight(square)
            ctrl = control_factor(self.board, square, color)
            score += w * ctrl
        return score

    def compute_shield_index(self, king_square, color=chess.WHITE):
        king_square_name = chess.square_name(king_square)
        king_file, king_rank = square_to_coord(king_square_name)
        shield_count = 0
        rank_offset = 1 if color == chess.WHITE else -1
        for df in [-1, 0, 1]:
            f = king_file + df
            r = king_rank + rank_offset
            if 0 <= f < 8 and 0 <= r < 8:
                square_str = chr(ord('a') + f) + str(r + 1)
                pawn_node = f"pawn_{square_str}"
                if self.graph.has_node(pawn_node):
                    shield_count += 1
        return shield_count

    def compute_attackers_proximity(self, king_square, color=chess.WHITE):
        king_coord = square_to_coord(chess.square_name(king_square))
        proximity = 0.0
        enemy_color = not color
        for square in self.board_squares:
            sq = chess.parse_square(square)
            piece = self.board.piece_at(sq)
            if piece is not None and piece.color == enemy_color:
                dist = manhattan_distance(king_coord, square_to_coord(square))
                proximity += 1.0 / (dist + 0.1)
        return proximity

    # --- Spectral Invariants ---
    def compute_spectral_invariants(self, color=None):
        spectral = {}
        L = nx.laplacian_matrix(self.graph, weight="weight").todense()
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues_sorted = sorted(eigenvalues.tolist())
        spectral["laplacian_spectrum"] = eigenvalues_sorted
        spectral["fiedler_value"] = eigenvalues_sorted[1] if len(eigenvalues_sorted) > 1 else None

        ev_centrality = nx.eigenvector_centrality(self.graph, weight="weight", max_iter=1000)
        spectral["eigenvector_centrality"] = ev_centrality
        spectral["average_eigenvector_centrality"] = np.mean(list(ev_centrality.values()))

        spectral["average_clustering"] = nx.average_clustering(self.graph, weight="weight")

        if color is not None:
            G_inf = self.compute_influence_subgraph_by_color(color)
            if G_inf.number_of_edges() > 0:
                L_inf = nx.laplacian_matrix(G_inf, weight="weight").todense()
                eigenvalues_inf = np.linalg.eigvalsh(L_inf)
                eigenvalues_inf_sorted = sorted(eigenvalues_inf.tolist())
                spectral["influence_fiedler_value"] = eigenvalues_inf_sorted[1] if len(eigenvalues_inf_sorted) > 1 else None
            else:
                spectral["influence_fiedler_value"] = None

            try:
                spectral["influence_global_efficiency"] = nx.global_efficiency(G_inf)
            except Exception:
                spectral["influence_global_efficiency"] = None

            if G_inf.number_of_nodes() > 0:
                degrees = dict(G_inf.degree())
                spectral["influence_avg_degree"] = sum(degrees.values()) / G_inf.number_of_nodes()
            else:
                spectral["influence_avg_degree"] = None

            try:
                closeness = nx.closeness_centrality(G_inf, distance='weight')
                spectral["influence_avg_closeness"] = np.mean(list(closeness.values()))
            except Exception:
                spectral["influence_avg_closeness"] = None

            try:
                betweenness = nx.betweenness_centrality(G_inf, weight='weight')
                spectral["influence_avg_betweenness"] = np.mean(list(betweenness.values()))
            except Exception:
                spectral["influence_avg_betweenness"] = None

            try:
                spectral["influence_avg_clustering"] = nx.average_clustering(G_inf, weight="weight")
            except Exception:
                spectral["influence_avg_clustering"] = None
        else:
            spectral["influence_fiedler_value"] = None
            spectral["influence_global_efficiency"] = None
            spectral["influence_avg_degree"] = None
            spectral["influence_avg_closeness"] = None
            spectral["influence_avg_betweenness"] = None
            spectral["influence_avg_clustering"] = None

        return spectral

    def compute_composite_invariants(self, color=chess.WHITE):
        king_sq = self.board.king(color)
        if king_sq is None:
            king_sq = chess.parse_square("e1") if color == chess.WHITE else chess.parse_square("e8")
        invariants = {
            "pawn_island_count": self.compute_pawn_island_count(),
            "passed_pawn_score": self.compute_passed_pawn_score(),
            "space_score": self.compute_space_score(color=color),
            "shield_index": self.compute_shield_index(king_sq, color=color),
            "attackers_proximity": self.compute_attackers_proximity(king_sq, color=color),
            "spectral_invariants": self.compute_spectral_invariants(color=color)
        }
        return invariants

    def compute_all_composite_invariants(self):
        return {
            "white": self.compute_composite_invariants(color=chess.WHITE),
            "black": self.compute_composite_invariants(color=chess.BLACK)
        }

    # ------------------------------------------------------------------------------
    # Output Table Function
    # ------------------------------------------------------------------------------
    def print_invariants_table(self, all_invariants):
        white = all_invariants["white"]
        black = all_invariants["black"]
        spec_white = white["spectral_invariants"]
        spec_black = black["spectral_invariants"]

        table = [
            ["Pawn Island Count", white["pawn_island_count"], black["pawn_island_count"]],
            ["Passed Pawn Score", white["passed_pawn_score"], black["passed_pawn_score"]],
            ["Space Score", white["space_score"], black["space_score"]],
            ["Shield Index", white["shield_index"], black["shield_index"]],
            ["Attackers' Proximity", white["attackers_proximity"], black["attackers_proximity"]],
            ["Fiedler Value", spec_white["fiedler_value"], spec_black["fiedler_value"]],
            ["Influence Fiedler Value", spec_white["influence_fiedler_value"], spec_black["influence_fiedler_value"]],
            ["Avg. Eigenvector Centrality", spec_white["average_eigenvector_centrality"], spec_black["average_eigenvector_centrality"]],
            ["Avg. Clustering", spec_white["average_clustering"], spec_black["average_clustering"]],
            ["Avg. Influence Clustering", spec_white["influence_avg_clustering"], spec_black["influence_avg_clustering"]],
            ["Influence Global Efficiency", spec_white["influence_global_efficiency"], spec_black["influence_global_efficiency"]],
            ["Influence Avg. Degree", spec_white["influence_avg_degree"], spec_black["influence_avg_degree"]],
            ["Influence Avg. Closeness", spec_white["influence_avg_closeness"], spec_black["influence_avg_closeness"]],
            ["Influence Avg. Betweenness", spec_white["influence_avg_betweenness"], spec_black["influence_avg_betweenness"]]
        ]
        col1_width = max(len(row[0]) for row in table)
        col2_width = max(len(str(row[1])) for row in table)
        col3_width = max(len(str(row[2])) for row in table)
        total_width = col1_width + col2_width + col3_width + 10
        print("-" * total_width)
        header = f"| {'Invariant':<{col1_width}} | {'White':^{col2_width}} | {'Black':^{col3_width}} |"
        print(header)
        print("-" * total_width)
        for row in table:
            print(f"| {row[0]:<{col1_width}} | {str(row[1]):^{col2_width}} | {str(row[2]):^{col3_width}} |")
        print("-" * total_width)

    # ------------------------------------------------------------------------------
    # Graph Visualization
    # ------------------------------------------------------------------------------
    def visualize(self, layout="spring"):
        if layout == "spring":
            pos = nx.spring_layout(self.graph, weight="weight")
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, weight="weight")
        node_colors = []
        for node, attr in self.graph.nodes(data=True):
            if attr.get("type") == "square":
                node_colors.append("lightblue")
            elif attr.get("type") == "pawn":
                node_colors.append("orange")
            elif attr.get("type") == "zone":
                node_colors.append("lightgreen")
            else:
                node_colors.append("gray")
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=300)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        edge_types = {
            "adjacency": "black",
            "pawn_support": "red",
            "influence": "purple",
            "zone": "green",
        }
        for etype, color in edge_types.items():
            edgelist = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == etype]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edgelist, edge_color=color, width=1)
        plt.title("Enhanced Positional Graph with Pawn Roles, Zone Nodes, and Spectral Invariants")
        plt.axis("off")
        plt.show()

# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Open Ruy Lopez / Active Middlegame (Example 5)
    fen = "4k3/8/8/8/8/2N1N3/8/3K4 w - - 0 1"
    board = chess.Board(fen)
    
    # Instantiate the PositionalGraph.
    pos_graph = PositionalGraph(board)
    
    # Compute composite invariants for both white and black.
    all_invariants = pos_graph.compute_all_composite_invariants()
    
    # Print the composite invariants in a formatted table.
    print("\nComposite Invariants Table:")
    pos_graph.print_invariants_table(all_invariants)
    
    # Visualize the overall positional graph.
    pos_graph.visualize(layout = "circular")
