# chess.py ♟️

An interactive chess game built with Python and Pygame. Play PvP, face a local AI, or run an Arena where two algorithms battle each other. The game tracks moves in the terminal and saves them to moves.txt.

## Table of Contents
- About
- Features
- Built With
- Getting Started
- Usage
- Arena Mode
- Controls
- Algorithms
- Project Structure
- Roadmap
- Contributing
- License
- Acknowledgments

## About
chess.py is a lightweight desktop chess app with full move legality (castling, en passant, promotions) plus a simple AI framework. It includes an Arena mode so you can compare algorithms head-to-head.

## Features
- Full chess rules enforcement with check, checkmate, stalemate, castling, en passant, and promotions
- Pygame UI with click-to-move and legal-move highlighting
- Terminal move tracking and persistent moves.txt log
- PvP, PvAI, and Arena (AI vs AI) modes
- Built-in algorithms: Minimax, Alpha-Beta Pruning, MCTS, Negamax
- Reset button to restart a game instantly

## Built With
- Python 3
- Pygame

## Getting Started

### Prerequisites
- Python 3.10+ recommended

### Installation
```bash
git clone https://github.com/your-username/chess.py.git
cd chess.py
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install pygame
```

## Usage
Run the game:

```bash
python chess_game.py
```

Moves are printed to the terminal and saved in moves.txt.

## Arena Mode
1. Toggle Arena: On.
2. Select the White and Black algorithms.
3. Click Confirm.
4. Watch the algorithms play each other, then review the victory overlay.

## Controls
- Click a piece, then click a destination square to move
- ESC cancels selection
- Promotion: press Q, R, B, or N when prompted
- Confirm locks mode and AI selections
- Reset restarts the game

## Algorithms
All algorithms prioritize immediate checkmate if available.
- Minimax
- Alpha-Beta Pruning
- Monte Carlo Tree Search (MCTS)
- Negamax

## Project Structure
- chess_game.py: game logic, UI, and mode handling
- minimax_ai.py: minimax search
- alphabeta_ai.py: alpha-beta search
- mcts_ai.py: Monte Carlo rollouts
- negamax_ai.py: negamax search
- moves.txt: move log output

## Roadmap
- Improve evaluation heuristics for stronger play
- Add selectable AI depth in the UI
- Add optional move undo

## Contributing
Pull requests are welcome. Please open an issue to discuss major changes first.

## License
Add a license to clarify usage and contributions.

## Acknowledgments
- Inspired by standard chess rules and classic board game UI patterns
