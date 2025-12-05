import pygame
import chess
import chess.engine
import random
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading

lock = threading.Lock()

# === Configuración visual ===
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
WHITE = (255, 255, 255)
GRAY = (125, 135, 150)
GREEN = (118, 150, 86)
FPS = 60

# Inicializar Pygame
pygame.init()
FONT = pygame.font.SysFont("Arial", 20)
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess vs DQN + Alpha-Beta")

# === Cargar imágenes de piezas de ajedrez ===
PIECE_IMAGES = {}
PIECES = ['P', 'N', 'B', 'R', 'Q', 'K']
for piece in PIECES:
    PIECE_IMAGES['w' + piece] = pygame.transform.scale(pygame.image.load(f"assets/w{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE))
    PIECE_IMAGES['b' + piece] = pygame.transform.scale(pygame.image.load(f"assets/b{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE))

# === Red neuronal para DQN ===
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(709, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# === Codificación del tablero y movimiento en un tensor ===
def board_to_tensor(board, move):
    board_fen = board.board_fen()
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
    }
    flat_board = [piece_map.get(c, 0) for c in board_fen.replace('/', '') if c.isalpha()]
    flat_board += [0] * (64 - len(flat_board))
    from_sq = move.from_square
    to_sq = move.to_square
    move_vec = [0]*645
    move_vec[from_sq] = 1
    move_vec[64 + to_sq] = 1
    move_vec[128 + (move.promotion if move.promotion else 0)] = 1
    return torch.tensor(flat_board + move_vec, dtype=torch.float32)

# === Evaluador para Alpha-Beta usando DQN ===
def evaluate_with_dqn(board, model):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return -999 if board.turn == chess.WHITE else 999

    qs = [model(board_to_tensor(board, move).unsqueeze(0)).item() for move in legal_moves]
    dqn_score = max(qs) if board.turn == chess.WHITE else -max(qs)

    classical_score = classical_evaluation(board)

    # Combinamos ambas evaluaciones
    combined_score = 0.8 * dqn_score + 0.2 * classical_score
    return combined_score

# === Alpha-Beta con evaluación por DQN ===
def alpha_beta(board, model, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_with_dqn(board, model), None

    best_move = None
    legal_moves = list(board.legal_moves)

    if maximizing:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = alpha_beta(board, model, depth - 1, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = alpha_beta(board, model, depth - 1, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# === Elegir movimiento con Alpha-Beta o DQN simple ===
def choose_action_with_alpha_beta(board, model, use_alpha_beta=True, depth=2):
    legal_moves = list(board.legal_moves)
    if use_alpha_beta:
        _, best_move = alpha_beta(board, model, depth, -float('inf'), float('inf'), maximizing=board.turn == chess.WHITE)
        if best_move:
            # Si es un peón que llega a la fila de promoción pero no tiene promotion asignado
            if best_move.promotion is None:
                if board.piece_at(best_move.from_square) and board.piece_at(best_move.from_square).piece_type == chess.PAWN and chess.square_rank(best_move.to_square) in [0, 7]:
                    return chess.Move(best_move.from_square, best_move.to_square, promotion=chess.QUEEN)
            elif best_move.promotion and best_move not in legal_moves:
                for move in legal_moves:
                    if (move.from_square == best_move.from_square and move.to_square == best_move.to_square and move.promotion == best_move.promotion):
                        return move
        return best_move
    else:
        best_move = None
        best_q = -float('inf')
        for move in legal_moves:
            state_tensor = board_to_tensor(board, move)
            q = model(state_tensor.unsqueeze(0)).item()
            if q > best_q:
                best_q = q
                best_move = move
        return best_move

# === Clase agente DQN extendida para entrenamiento ===
class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.discount = 0.95
        self.epsilon = 0.2
        if os.path.exists("dqn_model.pt"):
            self.model.load_state_dict(torch.load("dqn_model.pt"))
            self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, board):
        legal_moves = list(board.legal_moves)
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        best_q = -float('inf')
        best_move = None
        for move in legal_moves:
            state_tensor = board_to_tensor(board, move)
            q = self.model(state_tensor.unsqueeze(0)).item()
            if q > best_q:
                best_q = q
                best_move = move
        return best_move

    def calculate_reward(self, old_board, move, new_board):
        reward = 0.0
        if new_board.is_capture(move):
            captured_piece = old_board.piece_at(move.to_square)
            if captured_piece:
                reward += {chess.PAWN: 0.1, chess.KNIGHT: 0.3, chess.BISHOP: 0.3,
                           chess.ROOK: 0.5, chess.QUEEN: 0.9}.get(captured_piece.piece_type, 0)
        if new_board.is_check():
            reward += 0.05
        if new_board.is_check() and not old_board.is_check():
            reward += 0.2
        if move.promotion:
            reward += 0.8
        if new_board.is_checkmate():
            reward += 1.0 if new_board.turn == chess.BLACK else -1.0
        elif new_board.is_stalemate() or new_board.is_insufficient_material():
            reward += 0.3
        return reward

    def update(self, board, move, reward, next_board):
        self.model.train()
        state = board_to_tensor(board, move)
        legal_moves = list(next_board.legal_moves)
        if legal_moves:
            next_qs = [self.model(board_to_tensor(next_board, m).unsqueeze(0)).item() for m in legal_moves]
            max_future_q = max(next_qs)
        else:
            max_future_q = 0
        target_q = reward + self.discount * max_future_q
        pred_q = self.model(state.unsqueeze(0))
        loss = self.criterion(pred_q, torch.tensor([[target_q]]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.model.state_dict(), "dqn_model.pt")

    def reset_model(self):
        if os.path.exists("dqn_model.pt"):
            os.remove("dqn_model.pt")
        self.model = DQN()
        print("Model reseted.")

    def train(self, iterations):
        #print(f"[TRAIN] Iniciando entrenamiento por {iterations} iteraciones...")

        for episode in range(1, iterations + 1):

            board = chess.Board()
            move_counter = 0

            while not board.is_game_over():
                move_counter += 1

                if move_counter > 500:
                    print(f"[EPISODIO {episode}] Max limit of moves reached. Game finished.")
                    break

                move = self.choose_action(board)

                if move is None:
                    break

                old_board = board.copy()
                board.push(move)

                reward = self.calculate_reward(old_board, move, board)

                self.update(old_board, move, reward, board)

            # ---------- Recompensa final ----------
            result = board.result()
            print(f"[EPISODE {episode}] Final Result: {result}")

            if result == "1-0":
                final_reward = 1
            elif result == "0-1":
                final_reward = -1
            else:
                final_reward = 0

        self.save_model()
        print("[TRAIN] Model Saved.")

    def train_vs_stockfish(self, iterations, stockfish_path="./stockfish-windows-x86-64-avx2.exe"):
        print("Training against Stockfish...")
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        for _ in range(iterations):
            board = chess.Board()
            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    move = self.choose_action(board)
                    old_board = board.copy()
                    board.push(move)
                    reward = self.calculate_reward(old_board, move, board)
                    self.update(old_board, move, reward, board)
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)
        engine.quit()
        self.save_model()
        print("Training Finished.")

# === Juego principal ===
def main(use_alpha_beta=True, alpha_beta_depth=2):
    clock = pygame.time.Clock()
    run = True
    board = chess.Board()
    model = DQN()
    if os.path.exists("dqn_model.pt"):
        model.load_state_dict(torch.load("dqn_model.pt"))

    selected_square = None
    thinking = False

    def agent_move():
        nonlocal thinking
        with lock:
            board_copy = board.copy()  # Copia para búsqueda segura sin modificar el original
        move = choose_action_with_alpha_beta(board_copy, model, use_alpha_beta=use_alpha_beta, depth=alpha_beta_depth)
        print(f"Black Agent choose movement: {move}")
        if move:
            with lock:
                if board.piece_at(move.from_square).piece_type == chess.PAWN and chess.square_rank(move.to_square) == 0:
                    move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                board.push(move)
        thinking = False

    while run:
        clock.tick(FPS)
        with lock:
            WIN.fill(GRAY)
            draw_board(WIN, board)
            pygame.display.flip()

            if board.is_game_over():
                print("End of the Game:", board.result())
                pygame.time.wait(3000)
                run = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            with lock:
                if board.turn == chess.WHITE and event.type == pygame.MOUSEBUTTONDOWN:
                    square = get_square_under_mouse()
                    if square is not None:
                        if selected_square is None:
                            if board.piece_at(square) and board.piece_at(square).color == chess.WHITE:
                                selected_square = square
                        else:
                            piece = board.piece_at(selected_square)
                            if piece and piece.piece_type == chess.PAWN and chess.square_rank(square) == 7:
                                move = chess.Move(selected_square, square, promotion=chess.QUEEN)
                            else:
                                move = chess.Move(selected_square, square)

                            if move in board.legal_moves:
                                board.push(move)
                            selected_square = None

        with lock:
            if board.turn == chess.BLACK and not board.is_game_over() and not thinking:
                thinking = True
                threading.Thread(target=agent_move).start()

    pygame.quit()

# === Dibujar tablero y piezas ===
def draw_board(win, board):
    for y in range(8):
        for x in range(8):
            rect = pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(win, GREEN if (x + y) % 2 else WHITE, rect)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_prefix = 'w' if piece.color == chess.WHITE else 'b'
            img = PIECE_IMAGES[color_prefix + piece.symbol().upper()]
            x = chess.square_file(square)
            y = 7 - chess.square_rank(square)
            win.blit(img, (x * SQUARE_SIZE, y * SQUARE_SIZE))

# === Obtener casilla seleccionada por el mouse ===
def get_square_under_mouse():
    mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
    x, y = [int(v // SQUARE_SIZE) for v in mouse_pos]
    if 0 <= x <= 7 and 0 <= y <= 7:
        return chess.square(x, 7 - y)
    return None

# === Menú visual ===
def draw_text_center(surface, text, size, y_offset=0):
    font = pygame.font.SysFont("Arial", size)
    text_surface = font.render(text, True, (0, 0, 0))
    rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + y_offset))
    surface.blit(text_surface, rect)

def menu():
    clock = pygame.time.Clock()
    agent = DQNAgent()
    input_mode = False
    input_text = ''
    selected_train_mode = None

    while True:
        WIN.fill((200, 200, 200))
        draw_text_center(WIN, "Chess vs DQN", 36, -120)
        draw_text_center(WIN, "Press the Key to Play or Train:", 22, -80)
        draw_text_center(WIN, "D - Play with DQN", 20, -40)
        draw_text_center(WIN, "A - Play with DQN + Alpha-Beta (depth 2)", 20, -10)
        draw_text_center(WIN, "T - Play with DQN + Alpha-Beta (depth 3)", 20, 20)
        draw_text_center(WIN, "E - Train against it self", 20, 50)
        draw_text_center(WIN, "S - Train against Stockfish", 20, 80)
        draw_text_center(WIN, "R - Reset the Model", 20, 110)

        if input_mode:
            draw_text_center(WIN, f"Iterations: {input_text}", 20, 150)

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if input_mode:
                    if event.key == pygame.K_RETURN:
                        try:
                            iterations = int(input_text)
                            if selected_train_mode == 'auto':
                                agent.train(iterations)
                            elif selected_train_mode == 'stockfish':
                                agent.train_vs_stockfish(iterations)
                            input_mode = False
                            input_text = ''
                        except ValueError:
                            input_text = ''
                            input_mode = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                else:
                    if event.key == pygame.K_d:
                        main(use_alpha_beta=False)
                        return
                    elif event.key == pygame.K_a:
                        main(use_alpha_beta=True, alpha_beta_depth=2)
                        return
                    elif event.key == pygame.K_t:
                        main(use_alpha_beta=True, alpha_beta_depth=3)
                        return
                    elif event.key == pygame.K_e:
                        input_mode = True
                        selected_train_mode = 'auto'
                        input_text = ''
                        print("Training against it self...")
                    elif event.key == pygame.K_s:
                        input_mode = True
                        selected_train_mode = 'stockfish'
                        input_text = ''
                    elif event.key == pygame.K_r:
                        agent.reset_model()

def classical_evaluation(board):
    score = 0
    material_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.1,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = material_values.get(piece.piece_type, 0)
            score += value if piece.color == chess.WHITE else -value

            if piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE:
                    score += 0.05 * rank
                else:
                    score -= 0.05 * (7 - rank)

                    # 🗡️ Bonus si pieza controla casillas en el centro
                center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
                attacks = board.attacks(square)
                control_center = len(attacks & set(center_squares))
                bonus = 0.03 * control_center
                score += bonus if piece.color == chess.WHITE else -bonus

                # ⚔️ Bonus si pieza está desarrollada (caballos y alfiles no en la fila 1/8)
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if (piece.color == chess.WHITE and rank > 1) or (piece.color == chess.BLACK and rank < 6):
                        score += 0.1 if piece.color == chess.WHITE else -0.1

    return score

if __name__ == "__main__":
    menu()