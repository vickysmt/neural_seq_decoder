import torch
import torch.nn as nn

class FocalCTCLoss(nn.Module):
    def __init__(self, gamma=2.0, blank=0, reduction="mean"):
        super().__init__()
        self.ctc_loss = torch.nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: (T, batch_size, num_classes) - log-probabilities from model
        targets: (total_target_length) - concatenated target sequences
        input_lengths: (batch_size) - valid input lengths
        target_lengths: (batch_size) - valid target lengths
        """
        # Compute CTC Loss (without reduction)
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        # Convert log probabilities back to probabilities
        probs = torch.exp(log_probs)  # Shape: (T, batch_size, num_classes)

        # Extract target probabilities (P_t)
        with torch.no_grad():
            target_probs = torch.zeros_like(targets, dtype=torch.float32, device=log_probs.device)
            start = 0
            for i, length in enumerate(target_lengths):
                target_probs[start:start+length] = probs[:, i, targets[start:start+length]].mean(dim=0)
                start += length

        # Compute Focal Loss weighting (1 - P_t) ** gamma
        focal_weight = (1 - target_probs) ** self.gamma

        # print(f"loss shape: {loss.shape}")  # Should be (batch_size, max_target_length)
        # print(f"focal_weight shape: {focal_weight.shape}")  # Should match loss shape

        # Apply Focal Weight to CTC Loss
        focal_ctc_loss = focal_weight * loss.unsqueeze(1)

        # Apply reduction (mean, sum, or none)
        if self.reduction == "mean":
            return focal_ctc_loss.mean()
        elif self.reduction == "sum":
            return focal_ctc_loss.sum()
        else:
            return focal_ctc_loss  # Return per-example loss if "none"
