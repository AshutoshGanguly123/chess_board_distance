import chess
import networkx as nx

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

    # -------------------------------
    # Updated Strategic Graph (side-specific)
    # -------------------------------
    def _build_strategic_graph(self, position):
        """
        For the overall strategic graph we may use a default version.
        (But in experiments we call the side-specific version.)
        """
        return self._build_strategic_graph_side(position, None)

    def _build_strategic_graph_side(self, position, side):
        """
        Constructs a strategic graph. When side is not None, it differentiates
        white from black by including only pieces belonging to that side and
        only the king of that side. Fixed central landmarks are always included.
        When side is None, no filtering is done.
        """
        G = nx.Graph()
        board = position.board
        # Invariant 1: Always add the four central landmarks (even if unoccupied)
        central_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for sq in central_squares:
            G.add_node(sq, role="central", occupied=(board.piece_at(sq) is not None))
        if side is not None:
            # Invariant 2: Add only the king of the given side
            king_sq = board.king(side)
            if king_sq is not None:
                G.add_node(king_sq, role="king")
            # Invariant 3: Add any piece in the central region belonging to that side.
            for square in chess.SQUARES:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                if 2 <= file <= 5 and 2 <= rank <= 5:
                    piece = board.piece_at(square)
                    if piece and piece.color == side and square not in G:
                        G.add_node(square, role="central_piece")
        else:
            # When side is None, add kings from both sides and all central pieces.
            for color in [chess.WHITE, chess.BLACK]:
                king_sq = board.king(color)
                if king_sq is not None:
                    G.add_node(king_sq, role="king")
            for square in chess.SQUARES:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                if 2 <= file <= 5 and 2 <= rank <= 5:
                    if board.piece_at(square) and square not in G:
                        G.add_node(square, role="central_piece")
        # Connect nodes that are spatially adjacent (neighbors on the board)
        nodes = list(G.nodes())
        for i, sq1 in enumerate(nodes):
            for sq2 in nodes[i+1:]:
                if (abs(chess.square_file(sq1) - chess.square_file(sq2)) <= 1 and
                    abs(chess.square_rank(sq1) - chess.square_rank(sq2)) <= 1):
                    G.add_edge(sq1, sq2)
        return G

    def _build_hybrid_graph(self, position):
        control_graph = self._build_control_graph(position)
        threat_graph = self._build_threat_graph(position).to_undirected()
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

# For side-specific strategic graphs, you can also call:
def build_strategic_graph_side(position, side):
    """Wrapper for the side-specific strategic graph."""
    gc = GraphConstructionFunctor("strategic")
    return gc._build_strategic_graph_side(position, side)

def build_control_graph_side(position, side):
    """Construct a control graph for the given side."""
    import networkx as nx
    board = position.board
    G = nx.Graph()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == side:
            G.add_node(square, piece=piece.symbol())
            for target in board.attacks(square):
                target_piece = board.piece_at(target)
                if target_piece and target_piece.color == side:
                    G.add_edge(square, target)
    return G

def build_threat_graph_side(position, side):
    """Construct a threat graph for the given side."""
    import networkx as nx
    board = position.board
    G = nx.DiGraph()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == side:
            for target in board.attacks(square):
                victim = board.piece_at(target)
                if victim and victim.color != side:
                    G.add_edge(square, target)
    return G

def build_hybrid_graph_side(position, side):
    """Compose the side-specific control, threat, and strategic graphs."""
    import networkx as nx
    control_graph = build_control_graph_side(position, side)
    threat_graph = build_threat_graph_side(position, side).to_undirected()
    strategic_graph = build_strategic_graph_side(position, side)
    try:
        G = nx.compose_all([control_graph, threat_graph, strategic_graph])
    except Exception as e:
        print("Error composing side hybrid graphs:", e)
        G = nx.Graph()
    return G
