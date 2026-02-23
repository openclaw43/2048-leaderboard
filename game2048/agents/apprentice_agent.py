from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from game2048.agents import BaseAgent, register_agent
from game2048.agents.expectimax_agent import ExpectimaxAgent
from game2048.agents.mcts_agent import MCTSAgent
from game2048.game import Game2048


TRAINING_LOGS_DIR = Path(__file__).parent.parent.parent / ".training_logs"


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
    teacher: Optional[BaseAgent] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if teacher is None:
        teacher = ExpectimaxAgent(depth=2)
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


def _save_training_artifacts(
    history: dict[str, list[float]],
    sample_predictions: list[dict[str, Any]],
    log_dir: Path,
    stopped_early: bool,
    best_epoch: int,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    history_data = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_accuracy": history["val_accuracy"],
        "epochs_completed": len(history["train_loss"]),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_accuracy": max(history["val_accuracy"]),
    }
    with open(log_dir / "training_history.json", "w") as f:
        json.dump(history_data, f, indent=2)
    with open(log_dir / "sample_predictions.json", "w") as f:
        json.dump(sample_predictions, f, indent=2)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax1 = axes[0]
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax1.axvline(
            x=best_epoch + 1,
            color="g",
            linestyle="--",
            label=f"Best (epoch {best_epoch + 1})",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = axes[1]
        ax2.plot(
            epochs,
            [a * 100 for a in history["val_accuracy"]],
            "g-",
            label="Val Accuracy",
            linewidth=2,
        )
        ax2.axvline(
            x=best_epoch + 1,
            color="g",
            linestyle="--",
            label=f"Best (epoch {best_epoch + 1})",
        )
        ax2.axhline(
            y=max(history["val_accuracy"]) * 100, color="gray", linestyle=":", alpha=0.5
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if stopped_early:
            fig.suptitle(
                f"Training Curves (Early Stopped at Epoch {len(epochs)})",
                fontsize=12,
                y=1.02,
            )
        else:
            fig.suptitle(f"Training Curves ({len(epochs)} Epochs)", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(log_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass


def _display_sample_predictions(
    network: NeuralNetwork, X_val: np.ndarray, y_val: np.ndarray, num_samples: int = 5
) -> list[dict[str, Any]]:
    predictions = []
    indices = np.random.choice(len(X_val), min(num_samples, len(X_val)), replace=False)
    for idx in indices:
        pred = network.predict(X_val[idx])
        pred_move = np.argmax(pred)
        actual_move = np.argmax(y_val[idx])
        pred_label = decode_move(int(pred_move))
        actual_label = decode_move(int(actual_move))
        confidence = float(pred[pred_move])
        is_correct = bool(pred_move == actual_move)
        predictions.append(
            {
                "sample_idx": int(idx),
                "predicted": pred_label,
                "actual": actual_label,
                "confidence": round(confidence, 4),
                "correct": is_correct,
                "probabilities": {
                    "up": round(float(pred[0]), 4),
                    "down": round(float(pred[1]), 4),
                    "left": round(float(pred[2]), 4),
                    "right": round(float(pred[3]), 4),
                },
            }
        )
    return predictions


def train_network(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: list[int] = [128, 128, 128],
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 64,
    val_split: float = 0.2,
    verbose: bool = True,
    seed: Optional[int] = None,
    patience: int = 5,
    log_dir: Optional[Path] = None,
) -> tuple[NeuralNetwork, dict[str, Any]]:
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
    best_epoch = 0
    epochs_without_improvement = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    velocity_w = [np.zeros_like(w) for w in network.weights]
    velocity_b = [np.zeros_like(b) for b in network.biases]
    momentum = 0.9
    stopped_early = False
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
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        val_loss = 0.0
        correct = 0
        for i in range(len(X_val)):
            pred = network.predict(X_val[i])
            pred_move = np.argmax(pred)
            actual_move = np.argmax(y_val[i])
            if pred_move == actual_move:
                correct += 1
            val_loss += network.cross_entropy_loss(
                pred.reshape(-1, 1), y_val[i].reshape(-1, 1)
            )
        avg_val_loss = val_loss / len(X_val) if len(X_val) > 0 else 0
        val_acc = correct / len(X_val) if len(X_val) > 0 else 0
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)
        if verbose:
            marker = "*" if val_acc > best_val_acc else ""
            print(
                f"  Epoch {epoch + 1}/{epochs}: loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.2%}{marker}"
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [w.copy() for w in network.weights]
            best_biases = [b.copy() for b in network.biases]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"  Early stopping: no improvement for {patience} epochs")
                stopped_early = True
                break
    if best_weights is not None and best_biases is not None:
        network.weights = best_weights
        network.biases = best_biases
    sample_predictions = _display_sample_predictions(
        network, X_val, y_val, num_samples=5
    )
    if verbose:
        print("\n  Sample Predictions:")
        for p in sample_predictions:
            status = "OK" if p["correct"] else "WRONG"
            print(
                f"    [{status}] Predicted: {p['predicted']:5s} | "
                f"Actual: {p['actual']:5s} | Conf: {p['confidence']:.2%}"
            )
    result: dict[str, Any] = {
        "val_accuracy": best_val_acc,
        "train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "history": history,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "sample_predictions": sample_predictions,
    }
    if log_dir is not None:
        _save_training_artifacts(
            history, sample_predictions, log_dir, stopped_early, best_epoch
        )
    return network, result


@register_agent("apprentice")
class ApprenticeAgent(BaseAgent):
    network: Optional[NeuralNetwork]
    model_path: Optional[str]
    inference_times: list[float]

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_sizes: list[int] = [128, 128, 128],
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
        teacher: Optional[BaseAgent] = None,
        hidden_sizes: list[int] = [128, 128, 128],
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 64,
        val_split: float = 0.2,
        verbose: bool = True,
        seed: Optional[int] = None,
        patience: int = 5,
        save_logs: bool = True,
    ) -> dict[str, Any]:
        if teacher is None:
            teacher = ExpectimaxAgent(depth=2)
        if verbose:
            teacher_name = type(teacher).__name__
            print(f"Phase 1: Generating training data from {teacher_name}...")
        X, y = generate_training_data(
            num_samples=num_samples,
            teacher=teacher,
            verbose=verbose,
            seed=seed,
        )
        log_dir: Optional[Path] = None
        if save_logs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = TRAINING_LOGS_DIR / timestamp
            if verbose:
                print(f"Phase 2: Training neural network (logs: {log_dir})...")
        elif verbose:
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
            patience=patience,
            log_dir=log_dir,
        )
        if verbose:
            early_stop_info = " (early stopped)" if history.get("stopped_early") else ""
            print(
                f"Training complete{early_stop_info}. Best validation accuracy: {history['val_accuracy']:.2%}"
            )
            if log_dir:
                print(f"Diagnostics saved to: {log_dir}")
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
        teacher: Optional[BaseAgent] = None,
        hidden_sizes: list[int] = [128, 128, 128],
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 64,
        verbose: bool = True,
        seed: Optional[int] = None,
        patience: int = 5,
        save_logs: bool = True,
    ) -> "ApprenticeAgent":
        agent = cls(hidden_sizes=hidden_sizes)
        agent.train(
            num_samples=num_samples,
            teacher=teacher,
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            seed=seed,
            patience=patience,
            save_logs=save_logs,
        )
        return agent
