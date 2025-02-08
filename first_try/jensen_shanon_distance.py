import chess
import chess.engine
import math

###################################
# 1. Probability Distribution
###################################
def clamp_eval(eval_cp, lower=-2000, upper=2000):
    """
    Clamps the centipawn score to avoid extreme exponential results.
    """
    return max(lower, min(upper, eval_cp))

def compute_outcome_distribution(board: chess.Board, engine: chess.engine.SimpleEngine, depth=20):
    """
    Returns a 3-element list: [p_white_win, p_draw, p_black_win],
    derived from the engine's centipawn evaluation from White’s perspective.
    """
    # 1) Get engine evaluation in centipawns from White’s perspective
    info = engine.analyse(board, limit=chess.engine.Limit(depth=depth))
    # Mate scores are converted to large +/-
    eval_cp = info["score"].white().score(mate_score=100000)  

    # 2) Clamp the eval
    e = clamp_eval(eval_cp)

    # 3) Convert to probabilities
    # Let's define a logistic curve for White winning:
    # p_white ~ 1 / (1 + 10^(-e / 400))
    p_white = 1.0 / (1.0 + 10.0 ** (- e / 400.0))

    # Define a small draw probability that is higher if e is small
    # Just a simple Gaussian-ish shape around e=0
    p_draw = 0.1 * math.exp(-abs(e) / 400.0)  # tweak as you like

    # Remainder is p_black
    p_black = 1.0 - p_white - p_draw

    # Just in case numerical under/overflow
    if p_black < 0.0:
        p_black = 0.0
    if p_black > 1.0:
        p_black = 1.0

    # Re-normalize if needed
    total = p_white + p_draw + p_black
    if total == 0:
        return [0.33, 0.34, 0.33]
    return [p_white / total, p_draw / total, p_black / total]

###################################
# 2. Jensen-Shannon Divergence
###################################
def kl_divergence(p, q):
    """
    Kullback-Leibler divergence D_KL(P || Q).
    p, q are probability lists.
    """
    epsilon = 1e-12  # guard against log(0)
    return sum(
        p_i * math.log((p_i + epsilon) / (q_i + epsilon), 2)
        for p_i, q_i in zip(p, q)
        if p_i > 0
    )

def jensen_shannon_distance(p, q):
    """
    JS distance = sqrt(JS divergence).
    Where JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5(P+Q).
    p, q are probability lists (sum to 1).
    """
    m = [(p_i + q_i) / 2.0 for p_i, q_i in zip(p, q)]
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return math.sqrt(js_div)

###################################
# 3. Final Info Distance
###################################
def info_distance(board1: chess.Board, board2: chess.Board, engine: chess.engine.SimpleEngine, depth=20):
    """
    Computes an information-theoretic distance between board1 and board2,
    by comparing [p_white, p_draw, p_black] distributions.
    """
    dist1 = compute_outcome_distribution(board1, engine, depth=depth)
    dist2 = compute_outcome_distribution(board2, engine, depth=depth)
    return jensen_shannon_distance(dist1, dist2)

###################################
# 4. Demo
###################################
def main_demo_info_distance(board_a, board_b, engine_path="/opt/homebrew/bin/stockfish"):
    # 1) Create the engine
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    # # 2) Make two random boards
    # board_a = chess.Board()
    # board_b = chess.Board()

    # # Make some random moves in each
    # import random
    # for _ in range(5):
    #     if not board_a.is_game_over():
    #         move = random.choice(list(board_a.legal_moves))
    #         board_a.push(move)
    #     if not board_b.is_game_over():
    #         move = random.choice(list(board_b.legal_moves))
    #         board_b.push(move)

    # print("Board A FEN:", board_a.fen())
    # print("Board B FEN:", board_b.fen())

    # 3) Compute info distance
    dist = info_distance(board_a, board_b, engine, depth=20)
    # print(f"Information-based distance = {js_dist:.4f}")

    # # 4) Optional: see the distributions themselves
    # distA = compute_outcome_distribution(board_a, engine)
    # distB = compute_outcome_distribution(board_b, engine)
    # print("distA = [P(white_win), P(draw), P(black_win)] =", distA)
    # print("distB = [P(white_win), P(draw), P(black_win)] =", distB)

    # Shutdown engine
    engine.quit()
    return dist

# if __name__ == "__main__":
#     main_demo_info_distance()
