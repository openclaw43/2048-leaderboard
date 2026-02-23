from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from game2048.agents import BaseAgent, register_agent
from game2048.agents.mcts_agent import MCTSAgent
from game2048.game import Game2048


def encode_board(grid: list[list[int]]) -> np.ndarray:
    features = np.zeros(64, dtype=np.float32)
    for i in range(4):
        for j in range(4):
            val = grid[i][j]
            if val > 0:
                log_val = int(np.log2(val))
                features[i * 16 + j * 4 + min(log_val, 3)] = 1.0
                if log_val >= 4:
                    features[16 + i * 12 + j * 3 + min(log_val - 4, 2)] = 1.0
                    if log_val >= 7:
                        features[28 + i * 8 + j * 2 + min(log_val - 7, 1)] = 1.0
                        if log_val >= 9:
                            features[44 + i * 4 + j] = min(log_val - 9, 3) / 3.0
    return features


def encode_move(move: Optional[str]) -> int:
    mapping: dict[str, int] = {"up": 0, "down": 1, "left": 2, "right": 3}
    if move is None:
        return 0
    return mapping[move]


def decode_move(idx: int) -> str:
    return ["up", "down", "left", "right"][idx]


class NeuralNetwork:
    sizes: list[int]
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def __init__(self, sizes: list[int], seed: Optional[int] = None) -> None:
        self.sizes = sizes
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            self.weights.append(rng.standard_normal((sizes[i + 1], sizes[i])) * scale)
            self.biases.append(np.zeros((sizes[i + 1], 1), dtype=np.float32))

    def relu(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.maximum(0, x).astype(np.float32)
        return result

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.greater(x, 0).astype(np.float32)
        return result

    def softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(shifted)
        result: np.ndarray = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        return result

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [x]
        current = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ current + b
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = self.softmax(z)
            activations.append(current)
        return current, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        output, _ = self.forward(x.reshape(-1, 1))
        return output.flatten()

    def cross_entropy_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        eps = 1e-10
        return float(-np.mean(np.sum(targets * np.log(predictions + eps), axis=0)))

    def backward(
        self, x: np.ndarray, y: np.ndarray, lr: float = 0.001
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        output, activations = self.forward(x)
        loss = self.cross_entropy_loss(output, y)
        delta = output - y
        weight_grads: list[np.ndarray] = []
        bias_grads: list[np.ndarray] = []
        for i in range(len(self.weights) - 1, -1, -1):
            weight_grads.insert(0, delta @ activations[i].T)
            bias_grads.insert(0, delta)
            if i > 0:
                delta = (self.weights[i].T @ delta) * self.relu_derivative(
                    self.weights[i - 1] @ activations[i - 1] + self.biases[i - 1]
                    if i > 0
                    else activations[i]
                )
        return loss, weight_grads, bias_grads

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"sizes": self.sizes, "weights": self.weights, "biases": self.biases}, f
            )

    @classmethod
    def load(cls, path: str) -> "NeuralNetwork":
        with open(path, "rb") as f:
            data = pickle.load(f)
        nn = cls.__new__(cls)
        nn.sizes = data["sizes"]
        nn.weights = data["weights"]
        nn.biases = data["biases"]
        return nn


def generate_training_data(
    num_samples: int,
    mcts_simulations: int = 20,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    teacher = MCTSAgent(simulations=mcts_simulations)
    samples_per_game = 50
    num_games = max(1, num_samples // samples_per_game)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    total_samples = 0
    game_seed_start = rng.integers(0, 1000000)
    for game_idx in range(num_games):
        game = Game2048(seed=int(game_seed_start + game_idx))
        move_count = 0
        while not game.game_over and total_samples < num_samples:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            move = teacher.choose_move(game)
            if move is None:
                break
            X_list.append(encode_board(game.grid))
            y_onehot = np.zeros(4, dtype=np.float32)
            y_onehot[encode_move(move)] = 1.0
            y_list.append(y_onehot)
            game.move(move)
            move_count += 1
            total_samples += 1
        if verbose and (game_idx + 1) % 10 == 0:
            print(
                f"  Game {game_idx + 1}/{num_games}: {total_samples} samples collected"
            )
        if total_samples >= num_samples:
            break
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    if verbose:
        print(f"Generated {len(X)} training samples")
    return X, y


def train_network(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: list[int] = [128, 64],
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 64,
    val_split: float = 0.2,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> tuple[NeuralNetwork, dict[str, float]]:
    rng = np.random.default_rng(seed)
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    input_size = X.shape[1]
    output_size = 4
    sizes = [input_size] + hidden_sizes + [output_size]
    network = NeuralNetwork(sizes, seed=seed)
    best_val_acc = 0.0
    best_weights: Optional[list[np.ndarray]] = None
    best_biases: Optional[list[np.ndarray]] = None
    history: dict[str, float] = {"val_accuracy": 0.0, "train_loss": 0.0}
    avg_loss = 0.0
    velocity_w = [np.zeros_like(w) for w in network.weights]
    velocity_b = [np.zeros_like(b) for b in network.biases]
    momentum = 0.9
    for epoch in range(epochs):
        current_lr = lr * (0.95 ** (epoch // 10))
        perm = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        total_loss = 0.0
        num_batches = 0
        for i in range(0, len(X_train_shuffled), batch_size):
            X_batch = X_train_shuffled[i : i + batch_size]
            y_batch = y_train_shuffled[i : i + batch_size]
            batch_loss = 0.0
            weight_grads_sum = [np.zeros_like(w) for w in network.weights]
            bias_grads_sum = [np.zeros_like(b) for b in network.biases]
            for j in range(len(X_batch)):
                x = X_batch[j].reshape(-1, 1)
                yt = y_batch[j].reshape(-1, 1)
                loss, w_grads, b_grads = network.backward(x, yt, lr)
                batch_loss += loss
                for k in range(len(weight_grads_sum)):
                    weight_grads_sum[k] += w_grads[k]
                    bias_grads_sum[k] += b_grads[k]
            batch_size_actual = len(X_batch)
            for k in range(len(network.weights)):
                grad_w = weight_grads_sum[k] / batch_size_actual
                grad_b = bias_grads_sum[k] / batch_size_actual
                velocity_w[k] = momentum * velocity_w[k] - current_lr * grad_w
                velocity_b[k] = momentum * velocity_b[k] - current_lr * grad_b
                network.weights[k] += velocity_w[k]
                network.biases[k] += velocity_b[k]
            total_loss += batch_loss / batch_size_actual
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        correct = 0
        for i in range(len(X_val)):
            pred = network.predict(X_val[i])
            pred_move = np.argmax(pred)
            actual_move = np.argmax(y_val[i])
            if pred_move == actual_move:
                correct += 1
        val_acc = correct / len(X_val) if len(X_val) > 0 else 0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [w.copy() for w in network.weights]
            best_biases = [b.copy() for b in network.biases]
        if verbose and (epoch + 1) % 5 == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.2%}"
            )
    if best_weights is not None and best_biases is not None:
        network.weights = best_weights
        network.biases = best_biases
    history["val_accuracy"] = best_val_acc
    history["train_loss"] = avg_loss
    return network, history


@register_agent("apprentice")
class ApprenticeAgent(BaseAgent):
    network: Optional[NeuralNetwork]
    model_path: Optional[str]
    inference_times: list[float]

    def __init__(
        self, model_path: Optional[str] = None, hidden_sizes: list[int] = [128, 64]
    ) -> None:
        self.model_path = model_path
        self.inference_times = []
        default_model = Path(__file__).parent / "apprentice_model.pkl"
        if model_path:
            if Path(model_path).exists():
                self.network = NeuralNetwork.load(model_path)
            else:
                self.network = None
        elif default_model.exists():
            self.network = NeuralNetwork.load(str(default_model))
        else:
            self.network = None

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        if len(valid_moves) == 1:
            return valid_moves[0]
        if self.network is None:
            return valid_moves[0]
        start_time = time.perf_counter()
        features = encode_board(game.grid)
        predictions = self.network.predict(features)
        move_scores = {move: predictions[encode_move(move)] for move in valid_moves}
        best_move = valid_moves[0]
        best_score = move_scores[best_move]
        for move, score in move_scores.items():
            if score > best_score:
                best_score = score
                best_move = move
        elapsed = (time.perf_counter() - start_time) * 1000
        self.inference_times.append(elapsed)
        return best_move

    def get_avg_inference_time(self) -> float:
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def train(
        self,
        num_samples: int = 20000,
        mcts_simulations: int = 20,
        hidden_sizes: list[int] = [128, 64],
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 64,
        val_split: float = 0.2,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> dict[str, float]:
        if verbose:
            print("Phase 1: Generating training data from MCTS...")
        X, y = generate_training_data(
            num_samples=num_samples,
            mcts_simulations=mcts_simulations,
            verbose=verbose,
            seed=seed,
        )
        if verbose:
            print("Phase 2: Training neural network...")
        self.network, history = train_network(
            X,
            y,
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            val_split=val_split,
            verbose=verbose,
            seed=seed,
        )
        if verbose:
            print(
                f"Training complete. Best validation accuracy: {history['val_accuracy']:.2%}"
            )
        return history

    def save_model(self, path: str) -> None:
        if self.network is None:
            raise ValueError("No trained model to save")
        self.network.save(path)

    def load_model(self, path: str) -> bool:
        if Path(path).exists():
            self.network = NeuralNetwork.load(path)
            return True
        return False

    @classmethod
    def from_training(
        cls,
        num_samples: int = 20000,
        mcts_simulations: int = 20,
        hidden_sizes: list[int] = [128, 64],
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 64,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> "ApprenticeAgent":
        agent = cls(hidden_sizes=hidden_sizes)
        agent.train(
            num_samples=num_samples,
            mcts_simulations=mcts_simulations,
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            seed=seed,
        )
        return agent
