from typing import List, Optional, Callable


def select_move(
    state,
    legal_moves: List,
    generate_legal_moves: Callable,
    apply_move: Callable,
    evaluate_state: Callable,
    is_in_check: Callable,
    max_depth: int = 2,
) -> Optional:
    if not legal_moves:
        return None
    root_color = state.turn

    def is_immediate_checkmate(curr_state, move) -> bool:
        next_state = apply_move(curr_state, move)
        moves = generate_legal_moves(next_state, next_state.turn)
        return not moves and is_in_check(next_state, next_state.turn)

    for mv in legal_moves:
        if is_immediate_checkmate(state, mv):
            return mv

    def terminal_score(curr_state):
        moves = generate_legal_moves(curr_state, curr_state.turn)
        if moves:
            return None, moves
        if is_in_check(curr_state, curr_state.turn):
            return (-100000 if curr_state.turn == root_color else 100000), moves
        return 0, moves

    def alphabeta(curr_state, depth: int, alpha: int, beta: int, maximizing: bool) -> int:
        score, moves = terminal_score(curr_state)
        if score is not None:
            return score
        if depth == 0:
            return evaluate_state(curr_state, root_color)
        if maximizing:
            value = -10**9
            for mv in moves:
                value = max(value, alphabeta(apply_move(curr_state, mv), depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        value = 10**9
        for mv in moves:
            value = min(value, alphabeta(apply_move(curr_state, mv), depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    best_move = legal_moves[0]
    best_value = -10**9
    alpha = -10**9
    beta = 10**9
    for mv in legal_moves:
        value = alphabeta(apply_move(state, mv), max_depth - 1, alpha, beta, False)
        if value > best_value:
            best_value = value
            best_move = mv
        alpha = max(alpha, value)
    return best_move
