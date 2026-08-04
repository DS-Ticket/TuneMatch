"""
Sequence Model — LSTM-based autoplay predictor.

Given a user's last N songs, predict the next song.
This is what powers the "autoplay" feature and why it beats
Apple Music's static graph traversal.

Architecture:
  Song embeddings (32-dim) → LSTM (256 hidden) → Linear → Track probabilities

Training signal: next-song prediction on playlist sequences.
"""

import pickle
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader, Dataset


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PlaylistSequenceDataset(Dataset):
    """
    Converts playlist sequences into (context, target) pairs.
    context: last max_len track indices
    target:  next track index
    """
    def __init__(
        self,
        sequences: list[list[int]],
        max_len: int = 20,
    ):
        self.samples = []
        for seq in sequences:
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                context = seq[max(0, i - max_len):i]
                # Pad left with 0 if shorter than max_len
                context = [0] * (max_len - len(context)) + context
                self.samples.append((context, seq[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target,  dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class AutoplayLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))      # (batch, seq_len, emb_dim)
        output, _ = self.lstm(embedded)                  # (batch, seq_len, hidden_dim)
        last_hidden = output[:, -1, :]                   # take last timestep
        logits = self.fc(self.dropout(last_hidden))      # (batch, vocab_size)
        return logits


# ── Trainer ───────────────────────────────────────────────────────────────────

class SequenceModelTrainer:
    def __init__(self, config: dict):
        self.config  = config
        self.cfg     = config["models"]["sequence_model"]
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AutoplayLSTM] = None
        self.track_id_to_idx: dict = {}
        self.idx_to_track_id: dict = {}

    def _build_vocab(self, track_ids: list) -> None:
        unique = sorted(set(track_ids))
        self.track_id_to_idx = {tid: i + 1 for i, tid in enumerate(unique)}  # 0 = padding
        self.idx_to_track_id = {i + 1: tid for i, tid in enumerate(unique)}
        logger.info(f"Vocabulary size: {len(unique):,} tracks")

    def _encode_sequences(self, sequences: list[list]) -> list[list[int]]:
        encoded = []
        for seq in sequences:
            enc = [self.track_id_to_idx[t] for t in seq if t in self.track_id_to_idx]
            if len(enc) >= 2:
                encoded.append(enc)
        return encoded

    def fit(
        self,
        sequences: list[list],
        val_sequences: Optional[list[list]] = None,
    ) -> "SequenceModelTrainer":
        all_track_ids = [t for seq in sequences for t in seq]
        self._build_vocab(all_track_ids)

        vocab_size = len(self.track_id_to_idx) + 1  # +1 for padding

        self.model = AutoplayLSTM(
            vocab_size=vocab_size,
            embedding_dim=self.cfg["embedding_dim"],
            hidden_dim=self.cfg["hidden_dim"],
            num_layers=self.cfg["num_layers"],
            dropout=self.cfg["dropout"],
        ).to(self.device)

        enc_train = self._encode_sequences(sequences)
        train_ds  = PlaylistSequenceDataset(enc_train, self.cfg["max_sequence_len"])
        train_dl  = DataLoader(train_ds, batch_size=self.cfg["batch_size"], shuffle=True)

        val_dl = None
        if val_sequences:
            enc_val = self._encode_sequences(val_sequences)
            val_ds  = PlaylistSequenceDataset(enc_val, self.cfg["max_sequence_len"])
            val_dl  = DataLoader(val_ds, batch_size=self.cfg["batch_size"])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["learning_rate"])
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg["epochs"]
        )

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name="autoplay_lstm"):
            mlflow.log_params(self.cfg)

            for epoch in range(self.cfg["epochs"]):
                self.model.train()
                total_loss = 0.0
                for context, target in train_dl:
                    context, target = context.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(context)
                    loss   = criterion(logits, target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_dl)
                scheduler.step()

                if (epoch + 1) % 5 == 0:
                    log_msg = f"Epoch {epoch+1}/{self.cfg['epochs']} | train_loss={avg_loss:.4f}"
                    if val_dl:
                        val_loss = self._evaluate(val_dl, criterion)
                        log_msg += f" | val_loss={val_loss:.4f}"
                        mlflow.log_metric("val_loss", val_loss, step=epoch)
                    logger.info(log_msg)
                    mlflow.log_metric("train_loss", avg_loss, step=epoch)

        return self

    def _evaluate(self, dl: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for context, target in dl:
                context, target = context.to(self.device), target.to(self.device)
                logits = self.model(context)
                total += criterion(logits, target).item()
        return total / len(dl)

    def predict_next(
        self,
        history: list,
        n: int = 10,
        exclude: Optional[set] = None,
    ) -> list[tuple]:
        """
        Given a history of track IDs, return top-n predicted next tracks.
        Returns: [(track_id, probability), ...]
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        max_len = self.cfg["max_sequence_len"]
        context = [self.track_id_to_idx.get(t, 0) for t in history[-max_len:]]
        context = [0] * (max_len - len(context)) + context
        tensor  = torch.tensor([context], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor).squeeze(0)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        exclude_set = exclude or set()
        results = []
        for idx in np.argsort(probs)[::-1]:
            tid = self.idx_to_track_id.get(int(idx))
            if tid and tid not in exclude_set:
                results.append((tid, float(probs[idx])))
            if len(results) == n:
                break

        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "track_id_to_idx": self.track_id_to_idx,
            "idx_to_track_id": self.idx_to_track_id,
            "cfg": self.cfg,
        }
        torch.save(state, path)
        logger.info(f"Saved sequence model → {path}")

    def load(self, path: Path) -> "SequenceModelTrainer":
        state = torch.load(path, map_location=self.device)
        self.track_id_to_idx = state["track_id_to_idx"]
        self.idx_to_track_id = state["idx_to_track_id"]
        vocab_size = len(self.track_id_to_idx) + 1
        self.model = AutoplayLSTM(
            vocab_size=vocab_size,
            embedding_dim=state["cfg"]["embedding_dim"],
            hidden_dim=state["cfg"]["hidden_dim"],
            num_layers=state["cfg"]["num_layers"],
            dropout=state["cfg"]["dropout"],
        ).to(self.device)
        self.model.load_state_dict(state["model_state"])
        return self
