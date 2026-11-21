

import os
import random
import sys
import threading
from collections import deque
from queue import Queue, Empty

import chess
import chess.engine
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
FPS = 60
WHITE, GRAY, GREEN = (255, 255, 255), (125, 135, 150), (118, 150, 86)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_model_v3.pt")
BEST_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_model_v3_best.pt")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKFISH_PATH = os.path.join(BASE_DIR, "stockfish-windows-x86-64-avx2.exe")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIECE_VALUES = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.1,
                chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}
INPUT_SIZE = 64 * 12 + 64 * 2 + 5 + 1 + 4 + 64


def board_to_tensor(board, move):
    """Encode all 64 squares, the candidate move, turn and castling state."""
    values = np.zeros(INPUT_SIZE, dtype=np.float32)
    offset = 0
    # One-hot piece planes, indexed by square then piece/color.
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_index = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            values[offset + square * 12 + piece_index] = 1.0
    offset += 64 * 12
    values[offset + move.from_square] = 1.0
    offset += 64
    values[offset + move.to_square] = 1.0
    offset += 64
    if move.promotion:
        promotion_index = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2,
                           chess.QUEEN: 3}.get(move.promotion, 4)
        values[offset + promotion_index] = 1.0
    offset += 5
    values[offset] = 1.0 if board.turn == chess.WHITE else -1.0
    offset += 1
    castling_rights = (board.has_kingside_castling_rights(chess.WHITE),
                       board.has_queenside_castling_rights(chess.WHITE),
                       board.has_kingside_castling_rights(chess.BLACK),
                       board.has_queenside_castling_rights(chess.BLACK))
    for index, available in enumerate(castling_rights):
        values[offset + index] = float(available)
    offset += 4
    if board.ep_square is not None:
        values[offset + board.ep_square] = 1.0
    return torch.from_numpy(values)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(INPUT_SIZE, 384), nn.ReLU(),
                                    nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, 1))

    def forward(self, x):
        return self.layers(x)


def classical_evaluation(board):
    """Material and simple positional score from White's perspective."""
    score = 0.0
    center = {chess.D4, chess.D5, chess.E4, chess.E5}
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        score += sign * PIECE_VALUES[piece.piece_type]
        rank = chess.square_rank(square) if piece.color == chess.WHITE else 7 - chess.square_rank(square)
        if piece.piece_type == chess.PAWN:
            score += sign * 0.05 * rank
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            score += sign * 0.03 * len(board.attacks(square) & center)
            if rank > 1:
                score += sign * 0.1
    return score


def terminal_score(board):
    outcome = board.outcome()
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 10000.0 if outcome.winner == chess.WHITE else -10000.0


def evaluate_position(board, model):
    terminal = terminal_score(board)
    if terminal is not None:
        return terminal
    moves = list(board.legal_moves)
    if not moves:
        return 0.0
    with torch.no_grad():
        states = torch.stack([board_to_tensor(board, move) for move in moves]).to(DEVICE)
        q = model(states).squeeze(-1)
        best = float(q.max().item())
    # Model value is for the side to move; convert it to White's perspective.
    return classical_evaluation(board) + (best if board.turn == chess.WHITE else -best)


def ordered_moves(board):
    def priority(move):
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if board.is_en_passant(move):
                victim_value = PIECE_VALUES[chess.PAWN]
            else:
                victim_value = PIECE_VALUES[victim.piece_type] if victim else 0
            attacker = board.piece_at(move.from_square)
            score += 10 + 10 * victim_value - (PIECE_VALUES[attacker.piece_type] if attacker else 0)
        if move.promotion:
            score += 8 + PIECE_VALUES[move.promotion]
        return score
    return sorted(board.legal_moves, key=priority, reverse=True)


def alpha_beta(board, model, depth, alpha=-float("inf"), beta=float("inf"), maximizing=None,
               transposition=None, ply=0):
    """White-centric minimax with alpha-beta pruning."""
    if transposition is None:
        transposition = {}
    if board.is_game_over():
        score = terminal_score(board)
        if score:
            score -= ply if score > 0 else -ply
        return score or 0.0, None
    if depth <= 0:
        return evaluate_position(board, model), None
    key = (board.fen(), depth)
    if key in transposition:
        return transposition[key]
    original_alpha, original_beta = alpha, beta
    if maximizing is None:
        maximizing = board.turn == chess.WHITE
    best_move = None
    if maximizing:
        best = -float("inf")
        for move in ordered_moves(board):
            board.push(move)
            score, _ = alpha_beta(board, model, depth - 1, alpha, beta, False, transposition, ply + 1)
            board.pop()
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    else:
        best = float("inf")
        for move in ordered_moves(board):
            board.push(move)
            score, _ = alpha_beta(board, model, depth - 1, alpha, beta, True, transposition, ply + 1)
            board.pop()
            if score < best:
                best, best_move = score, move
            beta = min(beta, best)
            if alpha >= beta:
                break
    if original_alpha < best < original_beta:
        transposition[key] = (best, best_move)
    return best, best_move


def choose_action(board, model, use_alpha_beta=True, depth=2):
    moves = list(board.legal_moves)
    if not moves:
        return None
    if use_alpha_beta:
        return alpha_beta(board, model, depth)[1]
    with torch.no_grad():
        states = torch.stack([board_to_tensor(board, move) for move in moves]).to(DEVICE)
        return moves[int(model(states).squeeze(-1).argmax().item())]


class DQNAgent:
    def __init__(self):
        self.model = DQN().to(DEVICE)
        self.target_model = DQN().to(DEVICE)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.SmoothL1Loss()
        self.discount = 0.95
        self.epsilon = 0.2
        self.updates = 0
        self.replay = deque(maxlen=20_000)
        self.batch_size = 32
        self.best_eval_score = -1.0

    def choose_action(self, board):
        moves = list(board.legal_moves)
        if not moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(moves)
        with torch.no_grad():
            states = torch.stack([board_to_tensor(board, m) for m in moves]).to(DEVICE)
            return moves[int(self.model(states).squeeze(-1).argmax().item())]

    def calculate_reward(self, board, move):
        reward = 0.0
        if board.is_capture(move):
            victim = chess.PAWN if board.is_en_passant(move) else board.piece_at(move.to_square).piece_type
            reward += {chess.PAWN: 0.1, chess.KNIGHT: 0.3, chess.BISHOP: 0.3,
                       chess.ROOK: 0.5, chess.QUEEN: 0.9}.get(victim, 0.0)
        if move.promotion:
            reward += 0.8
        return reward

    def update(self, board, move, reward, next_board, terminal=False):
        """Learn a side-to-move value; the opponent's next value has opposite sign."""
        state = board_to_tensor(board, move)
        next_states = [] if terminal else [board_to_tensor(next_board, m) for m in next_board.legal_moves]
        self.replay.append((state, float(reward), next_states))
        if len(self.replay) < self.batch_size:
            return
        batch = random.sample(self.replay, self.batch_size)
        states = torch.stack([item[0] for item in batch]).to(DEVICE)
        targets = torch.tensor([item[1] for item in batch], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            for i, (_, _, candidates) in enumerate(batch):
                if candidates:
                    candidate_batch = torch.stack(candidates).to(DEVICE)
                    best_next = self.target_model(candidate_batch).squeeze(-1).max()
                    # next_board.turn is the opponent, so its value is from the
                    # opposite perspective to the player who made this move.
                    targets[i] -= self.discount * best_next
        self.model.train()
        prediction = self.model(states).squeeze(-1)
        loss = self.criterion(prediction, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()

    def finish_game(self, board, last_transition):
        if last_transition is None:
            return
        start, move = last_transition
        outcome = board.outcome()
        reward = 0.0 if outcome is None or outcome.winner is None else (1.0 if start.turn == outcome.winner else -1.0)
        self.update(start, move, reward, board, terminal=True)

    def save_model(self):
        torch.save(self.model.state_dict(), MODEL_PATH)

    def reset_model(self):
        self.model = DQN().to(DEVICE)
        self.target_model = DQN().to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.replay.clear()
        self.best_eval_score = -1.0
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        print("Modelo reiniciado.")

    def train(self, iterations):
        for episode in range(1, iterations + 1):
            board = chess.Board()
            last_transition = None
            for _ in range(500):
                if board.is_game_over():
                    break
                move = self.choose_action(board)
                if move is None:
                    break
                old_board = board.copy()
                reward = self.calculate_reward(board, move)
                board.push(move)
                if not board.is_game_over():
                    self.update(old_board, move, reward, board)
                last_transition = (old_board, move)
            self.finish_game(board, last_transition)
            self.epsilon = max(0.05, self.epsilon * 0.995)
            print(f"Partida {episode}: {board.result(claim_draw=True)}")
        self.save_model()
        print(f"Entrenamiento finalizado: {iterations} partidas completadas. Modelo guardado en {MODEL_PATH}.", flush=True)

    def train_vs_stockfish(self, iterations, stockfish_path=STOCKFISH_PATH):
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            for episode in range(1, iterations + 1):
                board = chess.Board()
                last_transition = None
                agent_color = chess.WHITE if episode % 2 == 1 else chess.BLACK
                for _ in range(500):
                    if board.is_game_over():
                        break
                    if board.turn == agent_color:
                        move = self.choose_action(board)
                        if move is None:
                            break
                        old = board.copy()
                        reward = self.calculate_reward(board, move)
                        board.push(move)
                        if not board.is_game_over():
                            self.update(old, move, reward, board)
                        last_transition = (old, move)
                    else:
                        result = engine.play(board, chess.engine.Limit(time=0.1))
                        board.push(result.move)
                self.finish_game(board, last_transition)
                self.epsilon = max(0.05, self.epsilon * 0.995)
                print(f"Partida {episode}: {board.result(claim_draw=True)}")
        finally:
            engine.quit()
        self.save_model()
        print(f"Entrenamiento contra Stockfish finalizado: {iterations} partidas completadas. Modelo guardado en {MODEL_PATH}.", flush=True)

    def evaluate_vs_stockfish(self, games=20, stockfish_path=STOCKFISH_PATH):
        """Evaluate without exploration or learning, alternating colors."""
        wins = draws = losses = 0
        previous_epsilon = self.epsilon
        self.epsilon = 0.0
        self.model.eval()
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            for episode in range(1, games + 1):
                board = chess.Board()
                agent_color = chess.WHITE if episode % 2 else chess.BLACK
                for _ in range(500):
                    if board.is_game_over():
                        break
                    if board.turn == agent_color:
                        move = self.choose_action(board)
                    else:
                        move = engine.play(board, chess.engine.Limit(time=0.1)).move
                    if move is None:
                        break
                    board.push(move)
                outcome = board.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    draws += 1
                elif outcome.winner == agent_color:
                    wins += 1
                else:
                    losses += 1
                print(f"Evaluación {episode}/{games}: {board.result(claim_draw=True)}")
        finally:
            engine.quit()
            self.epsilon = previous_epsilon
        score = (wins + 0.5 * draws) / games
        print(f"Resultado: {wins} victorias, {draws} tablas, {losses} derrotas; puntuación {score:.1%}.")
        if score > self.best_eval_score:
            self.best_eval_score = score
            best_path = MODEL_PATH.replace(".pt", "_best.pt")
            torch.save(self.model.state_dict(), best_path)
            print(f"Mejor modelo de esta sesión guardado en {best_path}.")


pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess vs DQN + Alpha-Beta")
PIECE_IMAGES = {}
for piece in "PNBRQK":
    for color in "wb":
        path = os.path.join(BASE_DIR, "assets", f"{color}{piece}.png")
        PIECE_IMAGES[color + piece] = pygame.transform.scale(pygame.image.load(path), (SQUARE_SIZE, SQUARE_SIZE))


def draw_board(surface, board):
    for y in range(8):
        for x in range(8):
            pygame.draw.rect(surface, GREEN if (x + y) % 2 else WHITE,
                             (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    for square, piece in board.piece_map().items():
        color = "w" if piece.color == chess.WHITE else "b"
        x, y = chess.square_file(square), 7 - chess.square_rank(square)
        surface.blit(PIECE_IMAGES[color + piece.symbol().upper()], (x * SQUARE_SIZE, y * SQUARE_SIZE))


def get_square_under_mouse():
    x, y = pygame.mouse.get_pos()
    file, rank = x // SQUARE_SIZE, y // SQUARE_SIZE
    return chess.square(file, 7 - rank) if 0 <= file < 8 and 0 <= rank < 8 else None


def move_coordinates(move):
    """Return a readable origin/destination pair, including promotions."""
    text = f"{chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)}"
    if move.promotion:
        text += f" (promoción a {chess.piece_name(move.promotion)})"
    return text


def main(use_alpha_beta=True, alpha_beta_depth=2):
    board = chess.Board()
    print("Nueva partida: Blancas (jugador) contra Negras (DQN).", flush=True)
    model = DQN().to(DEVICE)
    starting_model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    if os.path.exists(starting_model_path):
        model.load_state_dict(torch.load(starting_model_path, map_location=DEVICE, weights_only=True))
        print(f"Modelo cargado: {os.path.basename(starting_model_path)}", flush=True)
    model.eval()
    clock = pygame.time.Clock()
    selected = None
    result_queue = Queue()
    thinking = False
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == chess.WHITE and not thinking:
                square = get_square_under_mouse()
                if square is None:
                    selected = None
                elif selected is None:
                    piece = board.piece_at(square)
                    selected = square if piece and piece.color == chess.WHITE else None
                else:
                    candidates = [m for m in board.legal_moves if m.from_square == selected and m.to_square == square]
                    move = next((m for m in candidates if m.promotion == chess.QUEEN), candidates[0] if candidates else None)
                    if move:
                        san = board.san(move)
                        print(f"{board.fullmove_number}. Blancas: {move_coordinates(move)} ({san})", flush=True)
                        board.push(move)
                    selected = None
        if board.turn == chess.BLACK and not board.is_game_over() and not thinking:
            thinking = True
            snapshot = board.copy()
            def calculate(snapshot=snapshot):
                try:
                    result_queue.put(choose_action(snapshot, model, use_alpha_beta, alpha_beta_depth))
                except Exception as exc:
                    result_queue.put(exc)
            threading.Thread(target=calculate, daemon=True).start()
        if thinking:
            try:
                move = result_queue.get_nowait()
                thinking = False
                if isinstance(move, Exception):
                    print(f"Error al buscar movimiento: {move}")
                    running = False
                elif move in board.legal_moves:
                    san = board.san(move)
                    print(f"{board.fullmove_number}... Negras (DQN): {move_coordinates(move)} ({san})", flush=True)
                    board.push(move)
            except Empty:
                pass
        draw_board(WIN, board)
        pygame.display.flip()
        if board.is_game_over() and running:
            print("Fin de la partida:", board.result(claim_draw=True))
            pygame.time.wait(1500)
            running = False
    pygame.quit()


def draw_text_center(surface, text, size, y_offset=0):
    font = pygame.font.SysFont("Arial", size)
    image = font.render(text, True, (0, 0, 0))
    surface.blit(image, image.get_rect(center=(WIDTH // 2, HEIGHT // 2 + y_offset)))


def menu():
    agent = DQNAgent()
    input_mode, input_text, training_mode = False, "", None
    clock = pygame.time.Clock()
    while True:
        WIN.fill((200, 200, 200))
        for text, offset, size in [("Chess vs DQN", -120, 36), ("Presiona una tecla:", -80, 22),
                                   ("D - Jugar con DQN", -40, 20), ("A - DQN + Alpha-Beta (nivel 2)", -10, 20),
                                   ("T - DQN + Alpha-Beta (nivel 3)", 20, 20), ("E - Entrenar en autojuego", 50, 20),
                                   ("S - Entrenar contra Stockfish", 80, 20), ("V - Evaluar contra Stockfish", 110, 20),
                                   ("R - Reiniciar modelo", 140, 20)]:
            draw_text_center(WIN, text, size, offset)
        if input_mode:
            draw_text_center(WIN, f"Partidas: {input_text}", 20, 180)
        pygame.display.flip()
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type != pygame.KEYDOWN:
                continue
            if input_mode:
                if event.key == pygame.K_RETURN:
                    try:
                        count = int(input_text)
                        if count > 0:
                            if training_mode == "auto":
                                agent.train(count)
                            elif training_mode == "stockfish":
                                agent.train_vs_stockfish(count)
                            else:
                                agent.evaluate_vs_stockfish(count)
                    except (ValueError, OSError, chess.engine.EngineError) as exc:
                        print(f"No se pudo iniciar el entrenamiento: {exc}")
                    input_mode, input_text = False, ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isdigit():
                    input_text += event.unicode
            elif event.key == pygame.K_d:
                main(False)
            elif event.key == pygame.K_a:
                main(True, 2)
            elif event.key == pygame.K_t:
                main(True, 3)
            elif event.key == pygame.K_e:
                input_mode, input_text, training_mode = True, "", "auto"
            elif event.key == pygame.K_s:
                input_mode, input_text, training_mode = True, "", "stockfish"
            elif event.key == pygame.K_v:
                input_mode, input_text, training_mode = True, "", "evaluate"
            elif event.key == pygame.K_r:
                agent.reset_model()


if __name__ == "__main__":
    menu()
