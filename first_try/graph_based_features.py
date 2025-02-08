import chess
import numpy as np
from scipy.linalg import eig as dense_eig

###############################################################################
# A) Piece Weighting
###############################################################################
PIECE_WEIGHTS = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0  # or a larger value if you want to emphasize King
}

def piece_weight(piece: chess.Piece) -> float:
    """
    Returns a signed weight for the given piece:
      + for White, - for Black.
    """
    base_val = PIECE_WEIGHTS[piece.piece_type]
    return base_val if piece.color == chess.WHITE else -base_val

###############################################################################
# B) Building the Adjacency Matrix
###############################################################################
def build_adjacency_matrix(board: chess.Board) -> np.ndarray:
    """
    Build a 64x64 adjacency matrix A where:
      A[i, j] = piece_weight if the piece on square i has a legal move to j.
    i, j range over [0..63].
    """
    A = np.zeros((64, 64), dtype=np.float64)
    all_legal_moves = list(board.legal_moves)  # all legal moves from the current position
    
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        w = piece_weight(piece)
        for move in all_legal_moves:
            # if a legal move starts from 'square', set adjacency
            if move.from_square == square:
                A[square, move.to_square] = w
    return A

###############################################################################
# C) Spectral Graph Helpers
###############################################################################
def principal_eigenvector(A: np.ndarray):
    """
    Returns (lambda_max, eigenvector) for the largest eigenvalue (by magnitude).
    """
    vals, vecs = dense_eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    
    idx_max = np.argmax(vals)
    return vals[idx_max], vecs[:, idx_max]

def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    """
    Standard Laplacian: L = D - A, where D is diag of row sums.
    """
    degrees = np.sum(A, axis=1)
    return np.diag(degrees) - A

def fiedler_vector(L: np.ndarray):
    """
    Returns (lambda_2, fiedler_vec), where lambda_2 is the second-smallest eigenvalue.
    """
    vals, vecs = dense_eig(L)
    vals = np.real(vals)
    vecs = np.real(vecs)
    idx_sorted = np.argsort(vals)
    f_idx = idx_sorted[1]  # second smallest
    return vals[f_idx], vecs[:, f_idx]

def top_two_eigs(A: np.ndarray):
    """
    Returns ((lambda1, lambda2), (v1, v2)) for the largest & 2nd largest eigenvalues.
    """
    vals, vecs = dense_eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    
    # Sort descending by eigenvalue
    idx_sorted = np.argsort(vals)[::-1]
    lam1, lam2 = vals[idx_sorted[0]], vals[idx_sorted[1]]
    v1, v2 = vecs[:, idx_sorted[0]], vecs[:, idx_sorted[1]]
    return (lam1, lam2), (v1, v2)

def sign_vector(vec: np.ndarray) -> np.ndarray:
    """
    Return a vector of +1 / -1 / 0 corresponding to the signs of vec.
    """
    out = np.zeros_like(vec)
    out[vec > 0] = 1
    out[vec < 0] = -1
    return out

###############################################################################
# D) The Five Metrics
###############################################################################
def directional_influence_metric(A: np.ndarray) -> float:
    """
    DIM: Cosine similarity between principal eigenvector and an all-ones reference vector.
    """
    _, v1 = principal_eigenvector(A)
    v1_norm = np.linalg.norm(v1)
    if v1_norm == 0:
        return 0.0
    
    ref = np.ones_like(v1)
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        return 0.0
    
    return float(np.dot(v1, ref) / (v1_norm * ref_norm))

def strategic_partitioning_index(A: np.ndarray) -> float:
    """
    SPI: We'll do a simple measure of how 'balanced' the Fiedler partition is.
    We'll ignore a reference partition for simplicity.
    
    Steps:
      1) Compute the Fiedler vector f.
      2) Count how many squares fall into f >= 0 vs. f < 0.
      3) Return ratio of smaller side to total => if partition is balanced => near 0.5
    """
    L = laplacian_matrix(A)
    _, f = fiedler_vector(L)
    
    positive_count = np.sum(f >= 0)
    negative_count = np.sum(f < 0)
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    
    # ratio of smaller group to total
    smaller_side = min(positive_count, negative_count)
    spi_value = smaller_side / total
    return float(spi_value)

def dynamic_potential_asymmetry(A: np.ndarray) -> float:
    """
    DPA: correlation between top two eigenvectors (v1, v2).
    """
    (_, _), (v1, v2) = top_two_eigs(A)
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    
    corr_matrix = np.corrcoef(v1, v2)
    return float(corr_matrix[0, 1])  # correlation

def positional_stability_contrast(A: np.ndarray) -> float:
    """
    PSC: Hamming distance of the sign vectors of v1 and v2, 
         where v1, v2 are the top two eigenvectors.
    """
    (_, _), (v1, v2) = top_two_eigs(A)
    s1 = sign_vector(v1)
    s2 = sign_vector(v2)
    return float(np.sum(s1 != s2))

def cross_connectivity_balance(A1: np.ndarray, A2: np.ndarray) -> float:
    """
    CCB: We'll do a simple measure: 
         sqrt( sum( (sorted_eigs(A1) - sorted_eigs(A2))^2 ) ) 
    """
    vals1, _ = dense_eig(A1)
    vals2, _ = dense_eig(A2)
    
    vals1 = np.real(vals1)
    vals2 = np.real(vals2)
    
    vals1_sorted = np.sort(vals1)
    vals2_sorted = np.sort(vals2)
    
    min_len = min(len(vals1_sorted), len(vals2_sorted))
    dist_sq = np.sum((vals1_sorted[:min_len] - vals2_sorted[:min_len])**2)
    return float(np.sqrt(dist_sq))

###############################################################################
# E) Simple Demo with Two Boards
###############################################################################
def main():
    # Board 1: standard initial position
    board1 = chess.Board()
    A1 = build_adjacency_matrix(board1)
    
    # Board 2: a sample position from a known FEN
    fen_str = "r1bqkbnr/1ppp1ppp/p1n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4"
    board2 = chess.Board(fen=fen_str)
    A2 = build_adjacency_matrix(board2)
    
    # Compute metrics for board 1
    dim1 = directional_influence_metric(A1)
    spi1 = strategic_partitioning_index(A1)
    dpa1 = dynamic_potential_asymmetry(A1)
    psc1 = positional_stability_contrast(A1)
    
    # Compute metrics for board 2
    dim2 = directional_influence_metric(A2)
    spi2 = strategic_partitioning_index(A2)
    dpa2 = dynamic_potential_asymmetry(A2)
    psc2 = positional_stability_contrast(A2)
    
    # CCB across board1 and board2
    ccb_12 = cross_connectivity_balance(A1, A2)
    
    print("=== Board 1 (Standard Initial Position) ===")
    print(board1)
    print(f"DIM: {dim1:.4f}")
    print(f"SPI: {spi1:.4f}")
    print(f"DPA: {dpa1:.4f}")
    print(f"PSC: {psc1:.4f}")
    
    print("\n=== Board 2 (FEN) ===")
    print(board2)
    print(f"DIM: {dim2:.4f}")
    print(f"SPI: {spi2:.4f}")
    print(f"DPA: {dpa2:.4f}")
    print(f"PSC: {psc2:.4f}")
    
    print("\nCross-Connectivity Balance (Board 1 vs Board 2):")
    print(f"CCB: {ccb_12:.4f}")

if __name__ == "__main__":
    main()
