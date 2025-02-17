#!/usr/bin/env python
"""
influence_viz_dual.py

For each half-move (each FEN) in a PGN game, this script computes and plots the influence graphs 
for both White and Black. For each side, it collects every piece’s legal moves (ignoring if the destination is occupied)
and builds a bipartite graph with:
  - Piece nodes (labeled with the piece symbol and originating square, drawn larger)
  - Square nodes (drawn smaller)
  
The graphs are laid out using a spring layout with reduced optimal distance (k=0.4, iterations=100)
so that edge lengths are short and clusters are better separated. The "total diameter" (i.e. the sum of the 
diameters of all connected components) is computed for each graph and displayed (via a double‑headed arrow along the longest path from the largest component) 
and printed at the top right of each subplot. High‑quality images (300 dpi) are saved for every half-move.
"""

import os
import chess
import chess.pgn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch

try:
    from community.community_louvain import best_partition
except ImportError:
    best_partition = None

# ---------- Helper Functions ----------

def square_to_coord(sq_name):
    """Convert a chess square name (e.g., "e4") into numeric coordinates (0–7, 0–7)."""
    file = ord(sq_name[0]) - ord('a')
    rank = int(sq_name[1]) - 1
    return (file, rank)

def compute_piece_influences_for_color(board, color):
    """
    For the given board, compute a dictionary mapping each piece of the given color
    (True for White, False for Black) to the set of squares it can influence via legal moves.
    
    Returns a dict of the form:
      { "p_e4": { "piece": "N", "from": "e4", "influences": set(["f6", "g5", ...]) }, ... }
    """
    influence_dict = {}
    legal_moves = list(board.legal_moves)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color == color:
            sq_name = chess.square_name(square)
            piece_id = f"p_{sq_name}"
            influence_dict[piece_id] = {
                "piece": piece.symbol(),
                "from": sq_name,
                "influences": set()
            }
            for move in legal_moves:
                if move.from_square == square:
                    dest = chess.square_name(move.to_square)
                    influence_dict[piece_id]["influences"].add(dest)
    return influence_dict

def build_influence_graph(inf_dict):
    """
    Given an influence dictionary (for one color), build a bipartite graph where:
      - Each key (e.g. "p_e4") becomes a piece node (type "piece") at the coordinate of its "from" square.
      - Every destination square in its influence set becomes a square node (type "square").
      - An edge (weight 1) is added between the piece node and each influenced square.
    """
    G = nx.Graph()
    for piece_id, data in inf_dict.items():
        pos = square_to_coord(data["from"])
        base_color = "blue" if data["piece"].isupper() else "red"
        G.add_node(piece_id, pos=pos, type="piece", piece=data["piece"], color=base_color)
        for sq in data["influences"]:
            if not G.has_node(sq):
                G.add_node(sq, pos=square_to_coord(sq), type="square")
            G.add_edge(piece_id, sq, weight=1)
    return G

def get_sum_of_diameters(G):
    """
    Compute the total diameter, defined as the sum of the diameters of all connected components in G.
    Also returns the longest path (i.e. the path corresponding to the largest component's diameter)
    so that we can draw a representative double-headed arrow.
    """
    components = list(nx.connected_components(G))
    total_dia = 0
    largest_comp = None
    for comp in components:
        H = G.subgraph(comp)
        try:
            d = nx.diameter(H)
        except nx.NetworkXError:
            d = 0
        total_dia += d
        if largest_comp is None or len(comp) > len(largest_comp):
            largest_comp = comp
    longest_path = []
    if largest_comp is not None:
        H = G.subgraph(largest_comp)
        for src, subdict in dict(nx.all_pairs_shortest_path(H)).items():
            for tgt, path in subdict.items():
                if len(path) - 1 == nx.diameter(H):
                    longest_path = path
                    break
            if longest_path:
                break
    return total_dia, longest_path

def draw_influence_graph(G, ax, base_color):
    """
    Draw the influence graph G on ax using a spring layout with reduced optimal distance (k=0.4, iterations=100).
    Piece nodes are drawn larger (800) with bold sans-serif labels; square nodes are drawn smaller (400) with serif labels.
    
    If community detection (via Louvain) is available, node colors are set by cluster; otherwise, piece nodes use base_color
    and square nodes become gray if influenced by >1 piece.
    
    The "total diameter" (sum of diameters of all connected components) is computed and its value is printed at the top right.
    Additionally, a double-headed arrow is drawn along the longest path from the largest component.
    """
    pos = nx.spring_layout(G, weight='weight', seed=42, k=0.4, iterations=100)
    
    if best_partition is not None and G.number_of_nodes() > 0:
        partition = best_partition(G)
        max_comm = max(partition.values())
        cmap = plt.cm.get_cmap('Set2', max_comm+1)
        node_colors = [cmap(partition[node]) for node in G.nodes()]
    else:
        node_colors = []
        for node, data in G.nodes(data=True):
            if data["type"] == "piece":
                node_colors.append(base_color)
            else:
                node_colors.append("gray" if G.degree(node) > 1 else base_color)
    
    piece_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "piece"]
    square_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "square"]
    sizes = {n: (800 if G.nodes[n]["type"]=="piece" else 400) for n in G.nodes()}
    
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), ax=ax,
                           node_color=node_colors,
                           node_size=[sizes[n] for n in G.nodes()])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", width=2)
    
    # Draw labels for piece nodes: "piece_symbol (square)" using bold sans-serif.
    piece_labels = {}
    for n in piece_nodes:
        data = G.nodes[n]
        orig_sq = n.split("_")[1] if "_" in n else ""
        piece_labels[n] = f"{data['piece']}\n({orig_sq})"
    nx.draw_networkx_labels(G, pos, labels=piece_labels, font_size=10,
                            font_family="DejaVu Sans", font_weight="bold", ax=ax)
    
    # Draw labels for square nodes: just the square name in a serif font.
    square_labels = {n: n for n in square_nodes}
    nx.draw_networkx_labels(G, pos, labels=square_labels, font_size=8,
                            font_family="Times New Roman", ax=ax)
    
    # Compute the sum of diameters over all connected components and get the longest path from the largest component.
    total_dia, longest_path = get_sum_of_diameters(G)
    if longest_path and len(longest_path) > 1:
        path_edges = list(zip(longest_path, longest_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax,
                               edge_color="black", width=3, style="dashed")
        arrow = FancyArrowPatch(posA=pos[longest_path[0]], posB=pos[longest_path[-1]],
                                arrowstyle='<->', color='black', lw=2)
        ax.add_patch(arrow)
    ax.text(0.95, 0.95, f"Total Diameter: {total_dia}", transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5))
    
    ax.set_title("Influence Graph")
    ax.axis("off")

def main():
    # Set file paths (adjust as needed)
    pgn_path = '/Users/ashutoshganguly/Desktop/Research Papers/abid_ashutosh_papers/chess_board_distance/try_8/freestyle_chess_games/alphazero_vs_stockfish.pgn'
    output_dir = 'influence_viz_dual_alphazero_vs_stockfish'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(pgn_path) as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            print("No game found in PGN.")
            return
        
        board = game.board()
        fens = [board.fen()]
        moves_list = [None]
        for move in game.mainline_moves():
            board.push(move)
            fens.append(board.fen())
            moves_list.append(move.uci())
        
        # For each half-move, compute influence graphs for both White and Black and plot them side by side.
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            # Compute influence dictionary for White and Black
            white_inf_dict = compute_piece_influences_for_color(board, chess.WHITE)
            black_inf_dict = compute_piece_influences_for_color(board, chess.BLACK)
            
            G_white = build_influence_graph(white_inf_dict)
            G_black = build_influence_graph(black_inf_dict)
            
            fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
            fig.suptitle(f"Half-move {i}\nFEN: {fen}", fontsize=16)
            
            draw_influence_graph(G_white, axes[0], base_color="blue")
            axes[0].set_title("White Influence Graph")
            
            draw_influence_graph(G_black, axes[1], base_color="red")
            axes[1].set_title("Black Influence Graph")
            
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            fig_path = os.path.join(output_dir, f"half_move_{i:02d}.png")
            plt.savefig(fig_path, dpi=300)
            plt.close(fig)
            print(f"Saved visualization for half-move {i} to {fig_path}")

if __name__ == "__main__":
    main()
