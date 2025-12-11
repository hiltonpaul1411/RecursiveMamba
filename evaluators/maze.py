import torch
import numpy as np
from typing import Dict, Optional

class MazeEvaluator:
    """
    Robust evaluator for MAZE datasets.
    Handles both Logits (Float) and Token IDs (Int) to prevent broadcasting errors.
    """

    required_outputs = {"preds"}

    def __init__(self, data_path: str, eval_metadata, **kwargs):
        self.correct = 0
        self.total_tokens = 0
        self.seq_correct = 0
        self.num_sequences = 0  # Denominator for sequence accuracy

    def begin_eval(self):
        self.correct = 0
        self.total_tokens = 0
        self.seq_correct = 0
        self.num_sequences = 0

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Get the raw prediction tensor
        p_tensor = preds["preds"]

        # --- FIX 1: Double Argmax Prevention ---
        # If input is Float (Logits), use argmax to get tokens.
        # If input is Int/Long (already Tokens from generation), use as is.
        if p_tensor.is_floating_point():
            pred = p_tensor.argmax(dim=-1).cpu().numpy()  # (B, T)
        else:
            pred = p_tensor.cpu().numpy()                 # (B, T)

        target = batch["labels"].cpu().numpy()            # (B, T)

        # --- FIX 2: Shape Mismatch Safety ---
        # If generation is shorter/longer than target, crop to the minimum length
        # to prevent "operands could not be broadcast together"
        if pred.shape != target.shape:
            min_len = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_len]
            target = target[:, :min_len]

        # 1. Token-level accuracy
        # Create a mask where the target is valid (not -100)
        valid_mask = target != -100
        
        self.correct += np.sum((pred == target) & valid_mask)
        self.total_tokens += np.sum(valid_mask)

        # 2. Sequence-level accuracy (Exact Match)
        # We need to treat -100 spots as "automatically correct" so they don't break the check.
        # Logic: It is correct IF (tokens match) OR (target is ignore_index)
        token_matches_or_ignored = (pred == target) | (target == -100)
        
        # Check if ALL tokens in the row satisfy the condition
        row_is_perfect = np.all(token_matches_or_ignored, axis=1)
        
        self.seq_correct += np.sum(row_is_perfect)
        self.num_sequences += target.shape[0]  # Add batch size (B)

    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None):
        if rank != 0:
            return None

        # Avoid division by zero
        token_acc = 0.0
        if self.total_tokens > 0:
            token_acc = float(self.correct) / float(self.total_tokens)

        seq_acc = 0.0
        if self.num_sequences > 0:
            seq_acc = float(self.seq_correct) / float(self.num_sequences)

        return {
            "MAZE/token_acc": token_acc,
            "MAZE/seq_acc": seq_acc
        }