from dataclasses import dataclass
import threading
from typing import List, Optional, Tuple

import alphabeta_ai
import mcts_ai
import minimax_ai
import negamax_ai


FILES = "abcdefgh"
RANKS = "12345678"
MOVES_LOG_PATH = "moves.txt"


@dataclass(frozen=True)
class Move:
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    piece: Tuple[str, str]
    captured: Optional[Tuple[str, str]] = None
    promotion: Optional[str] = None
    is_castling: bool = False
    is_en_passant: bool = False


@dataclass
class GameState:
    board: List[List[Optional[Tuple[str, str]]]]
    turn: str
    castling_rights: dict
    en_passant: Optional[Tuple[int, int]]
    halfmove_clock: int
    fullmove_number: int
    move_history: List[str]


def create_initial_board() -> List[List[Optional[Tuple[str, str]]]]:
    empty = [None] * 8
    board = [empty[:] for _ in range(8)]
    board[0] = [("b", "R"), ("b", "N"), ("b", "B"), ("b", "Q"), ("b", "K"), ("b", "B"), ("b", "N"), ("b", "R")]
    board[1] = [("b", "P")] * 8
    board[6] = [("w", "P")] * 8
    board[7] = [("w", "R"), ("w", "N"), ("w", "B"), ("w", "Q"), ("w", "K"), ("w", "B"), ("w", "N"), ("w", "R")]
    return board


def initial_state() -> GameState:
    return GameState(
        board=create_initial_board(),
        turn="w",
        castling_rights={"K": True, "Q": True, "k": True, "q": True},
        en_passant=None,
        halfmove_clock=0,
        fullmove_number=1,
        move_history=[],
    )


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < 8 and 0 <= col < 8


def parse_square(text: str) -> Optional[Tuple[int, int]]:
    if len(text) != 2:
        return None
    file_char, rank_char = text[0], text[1]
    if file_char not in FILES or rank_char not in RANKS:
        return None
    col = FILES.index(file_char)
    row = 7 - RANKS.index(rank_char)
    return row, col


def square_to_str(pos: Tuple[int, int]) -> str:
    row, col = pos
    return f"{FILES[col]}{RANKS[7 - row]}"


def piece_to_char(piece: Tuple[str, str]) -> str:
    color, kind = piece
    return kind if color == "w" else kind.lower()


def render_board(board: List[List[Optional[Tuple[str, str]]]]) -> str:
    lines = []
    for row in range(8):
        rank = 8 - row
        row_pieces = []
        for col in range(8):
            piece = board[row][col]
            row_pieces.append(piece_to_char(piece) if piece else ".")
        lines.append(f"{rank} " + " ".join(row_pieces))
    lines.append("  " + " ".join(FILES))
    return "\n".join(lines)


def opponent(color: str) -> str:
    return "b" if color == "w" else "w"


def find_king(board: List[List[Optional[Tuple[str, str]]]], color: str) -> Optional[Tuple[int, int]]:
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece and piece[0] == color and piece[1] == "K":
                return r, c
    return None


def is_square_attacked(state: GameState, square: Tuple[int, int], by_color: str) -> bool:
    board = state.board
    row, col = square
    pawn_dir = -1 if by_color == "w" else 1
    for dc in (-1, 1):
        r, c = row + pawn_dir, col + dc
        if in_bounds(r, c):
            piece = board[r][c]
            if piece and piece[0] == by_color and piece[1] == "P":
                return True

    knight_moves = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]
    for dr, dc in knight_moves:
        r, c = row + dr, col + dc
        if in_bounds(r, c):
            piece = board[r][c]
            if piece and piece[0] == by_color and piece[1] == "N":
                return True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        while in_bounds(r, c):
            piece = board[r][c]
            if piece:
                if piece[0] == by_color and piece[1] in ("R", "Q"):
                    return True
                break
            r += dr
            c += dc

    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        while in_bounds(r, c):
            piece = board[r][c]
            if piece:
                if piece[0] == by_color and piece[1] in ("B", "Q"):
                    return True
                break
            r += dr
            c += dc

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if in_bounds(r, c):
                piece = board[r][c]
                if piece and piece[0] == by_color and piece[1] == "K":
                    return True
    return False


def is_in_check(state: GameState, color: str) -> bool:
    king_pos = find_king(state.board, color)
    if king_pos is None:
        return False
    return is_square_attacked(state, king_pos, opponent(color))


def generate_pseudo_legal_moves(state: GameState, color: str) -> List[Move]:
    board = state.board
    moves: List[Move] = []
    pawn_dir = -1 if color == "w" else 1
    start_row = 6 if color == "w" else 1
    promotion_row = 0 if color == "w" else 7

    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if not piece or piece[0] != color:
                continue
            kind = piece[1]

            if kind == "P":
                one_row = row + pawn_dir
                if in_bounds(one_row, col) and board[one_row][col] is None:
                    if one_row == promotion_row:
                        for promo in ("Q", "R", "B", "N"):
                            moves.append(Move((row, col), (one_row, col), piece, promotion=promo))
                    else:
                        moves.append(Move((row, col), (one_row, col), piece))
                    two_row = row + 2 * pawn_dir
                    if row == start_row and board[two_row][col] is None:
                        moves.append(Move((row, col), (two_row, col), piece))

                for dc in (-1, 1):
                    capture_row = row + pawn_dir
                    capture_col = col + dc
                    if not in_bounds(capture_row, capture_col):
                        continue
                    target = board[capture_row][capture_col]
                    if target and target[0] == opponent(color):
                        if capture_row == promotion_row:
                            for promo in ("Q", "R", "B", "N"):
                                moves.append(Move((row, col), (capture_row, capture_col), piece, captured=target, promotion=promo))
                        else:
                            moves.append(Move((row, col), (capture_row, capture_col), piece, captured=target))

                    if state.en_passant == (capture_row, capture_col):
                        captured_piece = board[row][capture_col]
                        moves.append(
                            Move(
                                (row, col),
                                (capture_row, capture_col),
                                piece,
                                captured=captured_piece,
                                is_en_passant=True,
                            )
                        )

            if kind == "N":
                for dr, dc in [
                    (-2, -1),
                    (-2, 1),
                    (-1, -2),
                    (-1, 2),
                    (1, -2),
                    (1, 2),
                    (2, -1),
                    (2, 1),
                ]:
                    r, c = row + dr, col + dc
                    if not in_bounds(r, c):
                        continue
                    target = board[r][c]
                    if target is None or target[0] == opponent(color):
                        moves.append(Move((row, col), (r, c), piece, captured=target))

            if kind in ("B", "R", "Q"):
                directions = []
                if kind in ("B", "Q"):
                    directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                if kind in ("R", "Q"):
                    directions += [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    while in_bounds(r, c):
                        target = board[r][c]
                        if target is None:
                            moves.append(Move((row, col), (r, c), piece))
                        else:
                            if target[0] == opponent(color):
                                moves.append(Move((row, col), (r, c), piece, captured=target))
                            break
                        r += dr
                        c += dc

            if kind == "K":
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        r, c = row + dr, col + dc
                        if not in_bounds(r, c):
                            continue
                        target = board[r][c]
                        if target is None or target[0] == opponent(color):
                            moves.append(Move((row, col), (r, c), piece, captured=target))

                if color == "w" and row == 7 and col == 4:
                    if state.castling_rights.get("K"):
                        if board[7][5] is None and board[7][6] is None:
                            if not is_square_attacked(state, (7, 4), "b") and not is_square_attacked(state, (7, 5), "b") and not is_square_attacked(state, (7, 6), "b"):
                                moves.append(Move((7, 4), (7, 6), piece, is_castling=True))
                    if state.castling_rights.get("Q"):
                        if board[7][1] is None and board[7][2] is None and board[7][3] is None:
                            if not is_square_attacked(state, (7, 4), "b") and not is_square_attacked(state, (7, 3), "b") and not is_square_attacked(state, (7, 2), "b"):
                                moves.append(Move((7, 4), (7, 2), piece, is_castling=True))

                if color == "b" and row == 0 and col == 4:
                    if state.castling_rights.get("k"):
                        if board[0][5] is None and board[0][6] is None:
                            if not is_square_attacked(state, (0, 4), "w") and not is_square_attacked(state, (0, 5), "w") and not is_square_attacked(state, (0, 6), "w"):
                                moves.append(Move((0, 4), (0, 6), piece, is_castling=True))
                    if state.castling_rights.get("q"):
                        if board[0][1] is None and board[0][2] is None and board[0][3] is None:
                            if not is_square_attacked(state, (0, 4), "w") and not is_square_attacked(state, (0, 3), "w") and not is_square_attacked(state, (0, 2), "w"):
                                moves.append(Move((0, 4), (0, 2), piece, is_castling=True))

    return moves


def apply_move(state: GameState, move: Move) -> GameState:
    board = [row[:] for row in state.board]
    from_row, from_col = move.from_pos
    to_row, to_col = move.to_pos
    piece = move.piece
    target = board[to_row][to_col]
    board[from_row][from_col] = None

    if move.is_en_passant:
        capture_row = from_row
        capture_col = to_col
        target = board[capture_row][capture_col]
        board[capture_row][capture_col] = None

    if move.is_castling:
        if to_col == 6:
            rook_from = (from_row, 7)
            rook_to = (from_row, 5)
        else:
            rook_from = (from_row, 0)
            rook_to = (from_row, 3)
        rook_piece = board[rook_from[0]][rook_from[1]]
        board[rook_from[0]][rook_from[1]] = None
        board[rook_to[0]][rook_to[1]] = rook_piece

    placed_piece = piece
    if move.promotion:
        placed_piece = (piece[0], move.promotion)
    board[to_row][to_col] = placed_piece

    castling_rights = dict(state.castling_rights)
    if piece[1] == "K":
        if piece[0] == "w":
            castling_rights["K"] = False
            castling_rights["Q"] = False
        else:
            castling_rights["k"] = False
            castling_rights["q"] = False
    if piece[1] == "R":
        if piece[0] == "w" and from_row == 7 and from_col == 0:
            castling_rights["Q"] = False
        if piece[0] == "w" and from_row == 7 and from_col == 7:
            castling_rights["K"] = False
        if piece[0] == "b" and from_row == 0 and from_col == 0:
            castling_rights["q"] = False
        if piece[0] == "b" and from_row == 0 and from_col == 7:
            castling_rights["k"] = False
    if target and target[1] == "R":
        if target[0] == "w" and to_row == 7 and to_col == 0:
            castling_rights["Q"] = False
        if target[0] == "w" and to_row == 7 and to_col == 7:
            castling_rights["K"] = False
        if target[0] == "b" and to_row == 0 and to_col == 0:
            castling_rights["q"] = False
        if target[0] == "b" and to_row == 0 and to_col == 7:
            castling_rights["k"] = False

    en_passant = None
    if piece[1] == "P" and abs(to_row - from_row) == 2:
        en_passant = ((from_row + to_row) // 2, from_col)

    halfmove_clock = state.halfmove_clock + 1
    if piece[1] == "P" or move.captured or move.is_en_passant:
        halfmove_clock = 0

    fullmove_number = state.fullmove_number
    if state.turn == "b":
        fullmove_number += 1

    return GameState(
        board=board,
        turn=opponent(state.turn),
        castling_rights=castling_rights,
        en_passant=en_passant,
        halfmove_clock=halfmove_clock,
        fullmove_number=fullmove_number,
        move_history=state.move_history[:],
    )


def generate_legal_moves(state: GameState, color: str) -> List[Move]:
    moves = []
    for move in generate_pseudo_legal_moves(state, color):
        next_state = apply_move(state, move)
        if not is_in_check(next_state, color):
            moves.append(move)
    return moves


def format_move(move: Move) -> str:
    base = f"{square_to_str(move.from_pos)}{square_to_str(move.to_pos)}"
    if move.promotion:
        base += move.promotion.lower()
    return base


def state_to_fen(state: GameState) -> str:
    rows = []
    for row in state.board:
        empty = 0
        parts = []
        for piece in row:
            if piece is None:
                empty += 1
            else:
                if empty:
                    parts.append(str(empty))
                    empty = 0
                parts.append(piece_to_char(piece))
        if empty:
            parts.append(str(empty))
        rows.append("".join(parts))
    board_part = "/".join(rows)
    turn_part = state.turn
    castle = ""
    for key in ("K", "Q", "k", "q"):
        if state.castling_rights.get(key):
            castle += key
    castle_part = castle if castle else "-"
    en_passant_part = square_to_str(state.en_passant) if state.en_passant else "-"
    return f"{board_part} {turn_part} {castle_part} {en_passant_part} {state.halfmove_clock} {state.fullmove_number}"


def clone_state(state: GameState) -> GameState:
    return GameState(
        board=[row[:] for row in state.board],
        turn=state.turn,
        castling_rights=dict(state.castling_rights),
        en_passant=state.en_passant,
        halfmove_clock=state.halfmove_clock,
        fullmove_number=state.fullmove_number,
        move_history=state.move_history[:],
    )


def evaluate_state(state: GameState, color: str) -> int:
    values = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
    score = 0
    for row in state.board:
        for piece in row:
            if piece:
                value = values[piece[1]]
                score += value if piece[0] == color else -value
    return score


def build_game_over_text(state: GameState, arena_mode: bool, arena_white_model: str, arena_black_model: str) -> str:
    if is_in_check(state, state.turn):
        winner_color = opponent(state.turn)
        if arena_mode:
            winner_algo = arena_white_model if winner_color == "w" else arena_black_model
            return f"Checkmate. {winner_algo} wins."
        winner = "White" if winner_color == "w" else "Black"
        return f"Checkmate. {winner} wins."
    return "Stalemate."


def ai_select_move(state: GameState, legal_moves: List[Move], algorithm: str) -> Optional[Move]:
    if not legal_moves:
        return None
    selectors = {
        "Minimax": minimax_ai.select_move,
        "Alpha-Beta Pruning": alphabeta_ai.select_move,
        "Monte Carlo Tree Search (MCTS)": mcts_ai.select_move,
        "Negamax Algorithm": negamax_ai.select_move,
    }
    selector = selectors.get(algorithm)
    if not selector:
        return None
    return selector(state, legal_moves, generate_legal_moves, apply_move, evaluate_state, is_in_check)


def build_move_history(moves: List[str]) -> str:
    chunks = []
    for index in range(0, len(moves), 2):
        number = index // 2 + 1
        white = moves[index]
        black = moves[index + 1] if index + 1 < len(moves) else ""
        if black:
            chunks.append(f"{number}. {white} {black}")
        else:
            chunks.append(f"{number}. {white}")
    return " | ".join(chunks)


def parse_move_input(text: str) -> Tuple[str, Optional[str]]:
    value = text.strip().lower()
    if value in {"quit", "exit"}:
        return "quit", None
    if value in {"o-o", "0-0"}:
        return "castle_k", None
    if value in {"o-o-o", "0-0-0"}:
        return "castle_q", None
    if len(value) in {4, 5}:
        return "move", value
    return "invalid", None


def resolve_move_input(state: GameState, move_text: str, legal_moves: List[Move]) -> Optional[Move]:
    if len(move_text) not in {4, 5}:
        return None
    from_square = parse_square(move_text[0:2])
    to_square = parse_square(move_text[2:4])
    if not from_square or not to_square:
        return None
    promotion = move_text[4] if len(move_text) == 5 else None
    if promotion:
        promotion = promotion.upper()
        if promotion not in {"Q", "R", "B", "N"}:
            return None

    candidates = []
    for move in legal_moves:
        if move.from_pos == from_square and move.to_pos == to_square:
            candidates.append(move)

    if not candidates:
        return None

    if len(candidates) == 1 and candidates[0].promotion and not promotion:
        default_move = candidates[0]
        return Move(
            from_pos=default_move.from_pos,
            to_pos=default_move.to_pos,
            piece=default_move.piece,
            captured=default_move.captured,
            promotion="Q",
            is_castling=default_move.is_castling,
            is_en_passant=default_move.is_en_passant,
        )

    if promotion:
        for move in candidates:
            if move.promotion == promotion:
                return move
        return None

    return candidates[0]


def resolve_castling(state: GameState, side: str, legal_moves: List[Move]) -> Optional[Move]:
    for move in legal_moves:
        if move.is_castling:
            if side == "king" and move.to_pos[1] == 6:
                return move
            if side == "queen" and move.to_pos[1] == 2:
                return move
    return None


def append_move_to_file(notation: str) -> None:
    with open(MOVES_LOG_PATH, "a", encoding="utf-8") as file_handle:
        file_handle.write(f"{notation}\n")


def select_move_for_ui(
    legal_moves: List[Move],
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    promotion: Optional[str],
) -> Optional[Move]:
    matches = [move for move in legal_moves if move.from_pos == from_pos and move.to_pos == to_pos]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if promotion:
        for move in matches:
            if move.promotion == promotion:
                return move
        return None
    for move in matches:
        if move.promotion == "Q":
            return move
    return matches[0]


def print_moves_to_terminal(history: List[str]) -> None:
    if history:
        print(f"Moves: {build_move_history(history)}")


def pygame_main() -> None:
    try:
        import pygame
    except Exception:
        print("pygame is required. Install it with: pip install pygame")
        return

    pygame.init()
    cell_size = 80
    board_pixels = cell_size * 8
    side_width = 240
    status_height = 80
    screen = pygame.display.set_mode((board_pixels + side_width, board_pixels + status_height))
    pygame.display.set_caption("Chess")
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 28)
    ui_font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    light_color = (240, 217, 181)
    dark_color = (181, 136, 99)
    select_color = (80, 160, 240)
    move_color = (120, 200, 120)

    state = initial_state()
    with open(MOVES_LOG_PATH, "w", encoding="utf-8"):
        pass
    selected: Optional[Tuple[int, int]] = None
    pending_promotion: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    legal_moves = generate_legal_moves(state, state.turn)
    game_over_text = ""
    mode_options = ["PvP", "PvAI"]
    model_options = ["Minimax", "Alpha-Beta Pruning", "Monte Carlo Tree Search (MCTS)", "Negamax Algorithm"]
    selected_mode = "PvP"
    selected_model = "Minimax"
    arena_mode = False
    arena_white_model = "Minimax"
    arena_black_model = "Alpha-Beta Pruning"
    mode_open = False
    model_open = False
    white_model_open = False
    black_model_open = False
    ai_error = ""
    ai_color = "b"
    ai_lock = threading.Lock()
    ai_running = False
    ai_result: Optional[Move] = None
    ai_result_turn: Optional[str] = None
    ai_epoch = 0
    mode_confirmed = False

    def start_ai_thread(algorithm: str, thinking_label: str) -> None:
        nonlocal ai_running, ai_result, ai_error, ai_result_turn, ai_epoch
        if ai_running:
            return
        snapshot_state = clone_state(state)
        snapshot_moves = legal_moves[:]
        snapshot_epoch = ai_epoch
        ai_running = True
        ai_error = thinking_label
        print(f"{thinking_label} algorithm={algorithm}, turn={snapshot_state.turn}, legal_moves={len(snapshot_moves)}")

        def worker() -> None:
            nonlocal ai_running, ai_result, ai_error, ai_result_turn, ai_epoch
            move = ai_select_move(snapshot_state, snapshot_moves, algorithm)
            with ai_lock:
                ai_running = False
                if snapshot_epoch == ai_epoch:
                    ai_result = move
                    ai_result_turn = snapshot_state.turn
                    if move is None:
                        ai_error = "AI move failed"
                        print("AI move failed")

        threading.Thread(target=worker, daemon=True).start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    selected = None
                    pending_promotion = None
                if pending_promotion and event.unicode.lower() in {"q", "r", "b", "n"}:
                    promo = event.unicode.upper()
                    chosen = select_move_for_ui(legal_moves, pending_promotion[0], pending_promotion[1], promo)
                    if chosen:
                        next_state = apply_move(state, chosen)
                        notation = format_move(chosen)
                        next_state.move_history.append(notation)
                        append_move_to_file(notation)
                        print_moves_to_terminal(next_state.move_history)
                        state = next_state
                        selected = None
                        pending_promotion = None
                        legal_moves = generate_legal_moves(state, state.turn)
                        if not legal_moves:
                            game_over_text = build_game_over_text(state, arena_mode, arena_white_model, arena_black_model)
                    continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if x >= board_pixels:
                    mode_rect = pygame.Rect(board_pixels + 20, 30, side_width - 40, 30)
                    model_rect = pygame.Rect(board_pixels + 20, 130, side_width - 40, 30)
                    white_model_rect = pygame.Rect(board_pixels + 20, 110, side_width - 40, 30)
                    black_model_rect = pygame.Rect(board_pixels + 20, 160, side_width - 40, 30)
                    confirm_rect = pygame.Rect(board_pixels + 20, 210, side_width - 40, 34)
                    arena_rect = pygame.Rect(board_pixels + 20, 260, side_width - 40, 34)
                    reset_rect = pygame.Rect(board_pixels + 20, 310, side_width - 40, 34)
                    handled = False
                    if mode_rect.collidepoint(x, y):
                        mode_open = not mode_open
                        model_open = False
                        white_model_open = False
                        black_model_open = False
                        handled = True
                        continue
                    if arena_mode:
                        if white_model_rect.collidepoint(x, y):
                            white_model_open = not white_model_open
                            black_model_open = False
                            mode_open = False
                            handled = True
                            continue
                        if black_model_rect.collidepoint(x, y):
                            black_model_open = not black_model_open
                            white_model_open = False
                            mode_open = False
                            handled = True
                            continue
                    else:
                        if model_rect.collidepoint(x, y):
                            model_open = not model_open
                            mode_open = False
                            handled = True
                            continue
                    if arena_rect.collidepoint(x, y):
                        arena_mode = not arena_mode
                        if arena_mode:
                            selected_mode = "PvAI"
                        mode_confirmed = False
                        ai_error = ""
                        mode_open = False
                        model_open = False
                        white_model_open = False
                        black_model_open = False
                        with ai_lock:
                            ai_result = None
                            ai_result_turn = None
                        ai_running = False
                        ai_epoch += 1
                        handled = True
                        continue
                    if reset_rect.collidepoint(x, y):
                        state = initial_state()
                        with open(MOVES_LOG_PATH, "w", encoding="utf-8"):
                            pass
                        selected = None
                        pending_promotion = None
                        legal_moves = generate_legal_moves(state, state.turn)
                        game_over_text = ""
                        ai_error = ""
                        with ai_lock:
                            ai_result = None
                            ai_result_turn = None
                        ai_running = False
                        ai_epoch += 1
                        handled = True
                        continue
                    if mode_open:
                        for index, option in enumerate(mode_options):
                            option_rect = pygame.Rect(board_pixels + 20, 30 + 30 * (index + 1), side_width - 40, 30)
                            if option_rect.collidepoint(x, y):
                                selected_mode = option
                                mode_open = False
                                model_open = False
                                white_model_open = False
                                black_model_open = False
                                ai_error = ""
                                mode_confirmed = False
                                handled = True
                                break
                    if model_open and not arena_mode:
                        for index, option in enumerate(model_options):
                            option_rect = pygame.Rect(board_pixels + 20, 130 + 30 * (index + 1), side_width - 40, 30)
                            if option_rect.collidepoint(x, y):
                                selected_model = option
                                model_open = False
                                mode_open = False
                                ai_error = ""
                                mode_confirmed = False
                                handled = True
                                break
                    if white_model_open and arena_mode:
                        for index, option in enumerate(model_options):
                            option_rect = pygame.Rect(board_pixels + 20, 110 + 30 * (index + 1), side_width - 40, 30)
                            if option_rect.collidepoint(x, y):
                                arena_white_model = option
                                white_model_open = False
                                black_model_open = False
                                mode_open = False
                                ai_error = ""
                                mode_confirmed = False
                                handled = True
                                break
                    if black_model_open and arena_mode:
                        for index, option in enumerate(model_options):
                            option_rect = pygame.Rect(board_pixels + 20, 160 + 30 * (index + 1), side_width - 40, 30)
                            if option_rect.collidepoint(x, y):
                                arena_black_model = option
                                black_model_open = False
                                white_model_open = False
                                mode_open = False
                                ai_error = ""
                                mode_confirmed = False
                                handled = True
                                break
                    if confirm_rect.collidepoint(x, y):
                        mode_confirmed = not mode_confirmed
                        ai_error = ""
                        with ai_lock:
                            ai_result = None
                            ai_result_turn = None
                        ai_running = False
                        ai_epoch += 1
                        handled = True
                    if not handled:
                        mode_open = False
                        model_open = False
                        white_model_open = False
                        black_model_open = False
                elif y < board_pixels and not game_over_text:
                    if not mode_confirmed:
                        continue
                    if arena_mode:
                        continue
                    if selected_mode == "PvAI" and state.turn == ai_color:
                        continue
                    col = x // cell_size
                    row = y // cell_size
                    if selected is None:
                        piece = state.board[row][col]
                        if piece and piece[0] == state.turn:
                            selected = (row, col)
                    else:
                        if selected == (row, col):
                            selected = None
                            pending_promotion = None
                            continue
                        chosen = select_move_for_ui(legal_moves, selected, (row, col), None)
                        if chosen and chosen.promotion:
                            pending_promotion = (selected, (row, col))
                        elif chosen:
                            next_state = apply_move(state, chosen)
                            notation = format_move(chosen)
                            next_state.move_history.append(notation)
                            append_move_to_file(notation)
                            print_moves_to_terminal(next_state.move_history)
                            state = next_state
                            selected = None
                            pending_promotion = None
                            legal_moves = generate_legal_moves(state, state.turn)
                            if not legal_moves:
                                game_over_text = build_game_over_text(state, arena_mode, arena_white_model, arena_black_model)
                        else:
                            piece = state.board[row][col]
                            if piece and piece[0] == state.turn:
                                selected = (row, col)
                            else:
                                selected = None
                                pending_promotion = None

        with ai_lock:
            if ai_result:
                allow_result = False
                if mode_confirmed and not game_over_text and not pending_promotion:
                    if arena_mode:
                        allow_result = True
                    elif selected_mode == "PvAI" and state.turn == ai_color:
                        allow_result = True
                if not allow_result:
                    ai_result = None
                    ai_result_turn = None

        if mode_confirmed and not pending_promotion and not game_over_text:
            if arena_mode:
                algorithm = arena_white_model if state.turn == "w" else arena_black_model
                chosen = None
                with ai_lock:
                    if ai_result and ai_result_turn == state.turn:
                        chosen = ai_result
                        ai_result = None
                        ai_result_turn = None
                    elif ai_result:
                        ai_result = None
                        ai_result_turn = None
                if chosen:
                    ai_error = ""
                    print(f"Arena move ({algorithm}): {format_move(chosen)}")
                    next_state = apply_move(state, chosen)
                    notation = format_move(chosen)
                    next_state.move_history.append(notation)
                    append_move_to_file(notation)
                    print_moves_to_terminal(next_state.move_history)
                    state = next_state
                    selected = None
                    pending_promotion = None
                    legal_moves = generate_legal_moves(state, state.turn)
                    if not legal_moves:
                        game_over_text = build_game_over_text(state, arena_mode, arena_white_model, arena_black_model)
                elif not ai_running:
                    start_ai_thread(algorithm, "Arena thinking...")
            elif selected_mode == "PvAI" and state.turn == ai_color:
                algorithm = selected_model
                chosen = None
                with ai_lock:
                    if ai_result and ai_result_turn == state.turn:
                        chosen = ai_result
                        ai_result = None
                        ai_result_turn = None
                    elif ai_result:
                        ai_result = None
                        ai_result_turn = None
                if chosen:
                    ai_error = ""
                    print(f"AI move: {format_move(chosen)}")
                    next_state = apply_move(state, chosen)
                    notation = format_move(chosen)
                    next_state.move_history.append(notation)
                    append_move_to_file(notation)
                    print_moves_to_terminal(next_state.move_history)
                    state = next_state
                    selected = None
                    pending_promotion = None
                    legal_moves = generate_legal_moves(state, state.turn)
                    if not legal_moves:
                        game_over_text = build_game_over_text(state, arena_mode, arena_white_model, arena_black_model)
                elif not ai_running:
                    start_ai_thread(algorithm, "AI thinking...")

        screen.fill((20, 20, 20))
        for row in range(8):
            for col in range(8):
                color = light_color if (row + col) % 2 == 0 else dark_color
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, color, rect)
                if selected == (row, col):
                    pygame.draw.rect(screen, select_color, rect, 4)
                for move in legal_moves:
                    if selected == move.from_pos and move.to_pos == (row, col):
                        pygame.draw.rect(screen, move_color, rect, 4)

                piece = state.board[row][col]
                if piece:
                    text = font.render(piece_to_char(piece), True, (10, 10, 10) if piece[0] == "b" else (240, 240, 240))
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

        if game_over_text and arena_mode:
            overlay = pygame.Surface((board_pixels, board_pixels), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))
            victory_text = font.render(game_over_text, True, (240, 240, 240))
            victory_rect = victory_text.get_rect(center=(board_pixels // 2, board_pixels // 2))
            screen.blit(victory_text, victory_rect)

        status_rect = pygame.Rect(0, board_pixels, board_pixels + side_width, status_height)
        pygame.draw.rect(screen, (30, 30, 30), status_rect)
        if game_over_text:
            status_text = game_over_text
        else:
            side = "White" if state.turn == "w" else "Black"
            status = "check" if is_in_check(state, state.turn) else "ok"
            status_text = f"{side} to move ({status})"
            if pending_promotion:
                status_text = "Promotion: press Q, R, B, or N"
        status_render = small_font.render(status_text, True, (220, 220, 220))
        screen.blit(status_render, (12, board_pixels + 22))

        panel_rect = pygame.Rect(board_pixels, 0, side_width, board_pixels)
        pygame.draw.rect(screen, (40, 40, 40), panel_rect)
        mode_rect = pygame.Rect(board_pixels + 20, 30, side_width - 40, 30)
        model_rect = pygame.Rect(board_pixels + 20, 130, side_width - 40, 30)
        white_model_rect = pygame.Rect(board_pixels + 20, 110, side_width - 40, 30)
        black_model_rect = pygame.Rect(board_pixels + 20, 160, side_width - 40, 30)
        confirm_rect = pygame.Rect(board_pixels + 20, 210, side_width - 40, 34)
        arena_rect = pygame.Rect(board_pixels + 20, 260, side_width - 40, 34)
        reset_rect = pygame.Rect(board_pixels + 20, 310, side_width - 40, 34)
        pygame.draw.rect(screen, (60, 60, 60), mode_rect)
        if arena_mode:
            pygame.draw.rect(screen, (60, 60, 60), white_model_rect)
            pygame.draw.rect(screen, (60, 60, 60), black_model_rect)
        else:
            pygame.draw.rect(screen, (60, 60, 60), model_rect)
        pygame.draw.rect(screen, (70, 110, 70) if mode_confirmed else (80, 80, 80), confirm_rect)
        pygame.draw.rect(screen, (100, 70, 120) if arena_mode else (80, 80, 80), arena_rect)
        pygame.draw.rect(screen, (120, 80, 80), reset_rect)
        screen.blit(ui_font.render(f"Mode: {selected_mode}", True, (230, 230, 230)), (mode_rect.x + 6, mode_rect.y + 6))
        if arena_mode:
            screen.blit(ui_font.render(f"White: {arena_white_model}", True, (230, 230, 230)), (white_model_rect.x + 6, white_model_rect.y + 6))
            screen.blit(ui_font.render(f"Black: {arena_black_model}", True, (230, 230, 230)), (black_model_rect.x + 6, black_model_rect.y + 6))
        else:
            screen.blit(ui_font.render(f"Model: {selected_model}", True, (230, 230, 230)), (model_rect.x + 6, model_rect.y + 6))
        screen.blit(ui_font.render("Edit" if mode_confirmed else "Confirm", True, (230, 230, 230)), (confirm_rect.x + 6, confirm_rect.y + 7))
        screen.blit(ui_font.render("Arena: On" if arena_mode else "Arena: Off", True, (230, 230, 230)), (arena_rect.x + 6, arena_rect.y + 7))
        screen.blit(ui_font.render("Reset", True, (230, 230, 230)), (reset_rect.x + 6, reset_rect.y + 7))
        if mode_open:
            for index, option in enumerate(mode_options):
                option_rect = pygame.Rect(board_pixels + 20, 30 + 30 * (index + 1), side_width - 40, 30)
                pygame.draw.rect(screen, (80, 80, 80), option_rect)
                screen.blit(ui_font.render(option, True, (240, 240, 240)), (option_rect.x + 6, option_rect.y + 6))
        if model_open and not arena_mode:
            for index, option in enumerate(model_options):
                option_rect = pygame.Rect(board_pixels + 20, 130 + 30 * (index + 1), side_width - 40, 30)
                pygame.draw.rect(screen, (80, 80, 80), option_rect)
                screen.blit(ui_font.render(option, True, (240, 240, 240)), (option_rect.x + 6, option_rect.y + 6))
        if white_model_open and arena_mode:
            for index, option in enumerate(model_options):
                option_rect = pygame.Rect(board_pixels + 20, 110 + 30 * (index + 1), side_width - 40, 30)
                pygame.draw.rect(screen, (80, 80, 80), option_rect)
                screen.blit(ui_font.render(option, True, (240, 240, 240)), (option_rect.x + 6, option_rect.y + 6))
        if black_model_open and arena_mode:
            for index, option in enumerate(model_options):
                option_rect = pygame.Rect(board_pixels + 20, 160 + 30 * (index + 1), side_width - 40, 30)
                pygame.draw.rect(screen, (80, 80, 80), option_rect)
                screen.blit(ui_font.render(option, True, (240, 240, 240)), (option_rect.x + 6, option_rect.y + 6))
        if ai_error:
            screen.blit(ui_font.render(ai_error, True, (220, 120, 120)), (board_pixels + 20, 350))

        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    pygame_main()


if __name__ == "__main__":
    main()
