# chess_analysis/chess_position.py
import chess

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
