import random
from typing import List, Optional, Callable


def select_move(
    state,
    legal_moves: List,
    generate_legal_moves: Callable,
    apply_move: Callable,
    evaluate_state: Callable,
    is_in_check: Callable,
    iterations: int = 200,
    playout_depth: int = 6,
) -> Optional:
    if not legal_moves:
        return None
    root_color = state.turn
    totals = {move: 0 for move in legal_moves}
    counts = {move: 0 for move in legal_moves}

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

    def rollout(curr_state):
        score, moves = terminal_score(curr_state)
        if score is not None:
            return score
        depth = 0
        while depth < playout_depth:
            if not moves:
                if is_in_check(curr_state, curr_state.turn):
                    return -100000 if curr_state.turn == root_color else 100000
                return 0
            mv = random.choice(moves)
            curr_state = apply_move(curr_state, mv)
            score, moves = terminal_score(curr_state)
            if score is not None:
                return score
            depth += 1
        return evaluate_state(curr_state, root_color)

    for _ in range(iterations):
        mv = random.choice(legal_moves)
        score = rollout(apply_move(state, mv))
        totals[mv] += score
        counts[mv] += 1

    best_move = legal_moves[0]
    best_value = -10**9
    for mv in legal_moves:
        count = counts[mv]
        avg = totals[mv] / count if count else -10**9
        if avg > best_value:
            best_value = avg
            best_move = mv
    return best_move
