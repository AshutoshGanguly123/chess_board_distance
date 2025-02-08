# chess_analysis/evaluation.py
from stockfish import Stockfish

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
