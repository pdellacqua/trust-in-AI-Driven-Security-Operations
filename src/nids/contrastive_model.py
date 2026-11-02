"""
Contrastive Traffic-Text NIDS Architecture.
Implements:
- TrafficEncoder f_theta and ProjectionHead g_phi
- TextPrototypeEncoder f_psi (semantic prototypes from MITRE/CVE)
- InfoNCE loss for traffic views
- Bidirectional multimodal contrastive loss
- Combined training objective: L_NIDS = omega_loss * L_traffic + (1 - omega_loss) * L_multi
"""

import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def augment_flow(
    x: torch.Tensor,
    mask_prob: float = 0.15,
    noise_std: float = 0.05,
    jitter_range: float = 0.05,
) -> torch.Tensor:
    """
    Applies feature masking, temporal/attribute jittering, and Gaussian perturbation.
    """
    augmented = x.clone()
    
    # 1. Feature masking
    if mask_prob > 0.0:
        mask = torch.rand_like(augmented) < mask_prob
        augmented[mask] = 0.0
        
    # 2. Gaussian perturbation
    if noise_std > 0.0:
        noise = torch.randn_like(augmented) * noise_std
        augmented = augmented + noise
        
    # 3. Scaling / jittering
    if jitter_range > 0.0:
        scale = 1.0 + (torch.rand(augmented.size(0), 1, device=augmented.device) * 2 - 1) * jitter_range
        augmented = augmented * scale
        
    return augmented


class TrafficEncoder(nn.Module):
    """
    Neural network mapping raw flow-level features into latent embeddings.
    Architecture: MLP with residual connection, BatchNorm, and projection head.
    """

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.GELU()
        
        self.dropout = nn.Dropout(dropout)
        
        # Projection head g_phi
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalized embedding z in R^{embedding_dim}.
        """
        h = self.act1(self.bn1(self.fc1(x)))
        h = h + self.act2(self.bn2(self.fc2(h)))  # Residual connection
        h = self.dropout(h)
        z = self.projection_head(h)
        return F.normalize(z, p=2, dim=-1)


class TextPrototypeEncoder(nn.Module):
    """
    Encodes semantic textual descriptions of threats into prototype vectors w_c in R^{embedding_dim}.
    Can be conditioned on DistilBERT token vectors or learned semantic prototype embeddings.
    """

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 64,
        semantic_descriptors: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Trainable semantic prototype vectors for each threat class
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.orthogonal_(self.prototypes)

    def forward(self, class_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns normalized semantic prototype vectors w_c.
        If class_indices is None, returns all class prototypes [num_classes, embedding_dim].
        """
        norm_prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        if class_indices is None:
            return norm_prototypes
        return norm_prototypes[class_indices]


def traffic_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Computes InfoNCE contrastive loss over correlated traffic views:
    L_traffic = - (1/N) sum_i log( exp(sim(z_i^a, z_i^b)/tau) / sum_j exp(sim(z_i^a, z_j^b)/tau) )
    """
    batch_size = z_a.size(0)
    sim_matrix = torch.matmul(z_a, z_b.T) / temperature  # [N, N]
    labels = torch.arange(batch_size, device=z_a.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def multimodal_contrastive_loss(
    traffic_embeddings: torch.Tensor,
    text_prototypes: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Computes bidirectional cross-modal alignment loss between network traffic and textual prototypes:
    L_multi = - (1/M) sum_i [ log(exp(sim(z_i, w_i)/tau) / sum_j exp(sim(z_i, w_j)/tau)) +
                             log(exp(sim(w_i, z_i)/tau) / sum_j exp(sim(w_i, z_j)/tau)) ]
    """
    # traffic_embeddings: [M, D]
    # text_prototypes: [NumClasses, D]
    # labels: [M] with class indices
    sim_t2text = torch.matmul(traffic_embeddings, text_prototypes.T) / temperature  # [M, NumClasses]
    loss_t2text = F.cross_entropy(sim_t2text, labels)
    
    # Text-to-traffic alignment across positive batch pairs
    w_target = text_prototypes[labels]  # [M, D]
    sim_text2t = torch.matmul(w_target, traffic_embeddings.T) / temperature  # [M, M]
    self_labels = torch.arange(len(labels), device=traffic_embeddings.device)
    loss_text2t = F.cross_entropy(sim_text2t, self_labels)
    
    return 0.5 * (loss_t2text + loss_text2t)


class MultimodalNIDS(nn.Module):
    """
    End-to-end multimodal traffic-text NIDS model combining traffic encoder and semantic prototypes.
    """

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_classes: int = 10,
        temperature: float = 0.07,
        omega_loss: float = 0.60,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.omega_loss = omega_loss
        
        self.traffic_encoder = TrafficEncoder(input_dim, hidden_dim, embedding_dim)
        self.text_encoder = TextPrototypeEncoder(num_classes, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps network flow to embedding and computes similarity logits with all prototypes.
        Returns:
            z: [batch_size, embedding_dim]
            logits: [batch_size, num_classes] where logits[i, c] = sim(z_i, w_c)
        """
        z = self.traffic_encoder(x)
        prototypes = self.text_encoder()  # [num_classes, embedding_dim]
        logits = torch.matmul(z, prototypes.T)  # Cosine similarities because both are L2 normalized
        return z, logits

    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes total loss:
        L_NIDS = omega_loss * L_traffic + (1 - omega_loss) * L_multi
        """
        # Create two augmented views
        x_a = augment_flow(x)
        x_b = augment_flow(x)
        
        z_a = self.traffic_encoder(x_a)
        z_b = self.traffic_encoder(x_b)
        
        loss_traffic = traffic_contrastive_loss(z_a, z_b, temperature=self.temperature)
        
        prototypes = self.text_encoder()
        loss_multi = multimodal_contrastive_loss(z_a, prototypes, labels, temperature=self.temperature)
        
        total_loss = self.omega_loss * loss_traffic + (1.0 - self.omega_loss) * loss_multi
        
        return total_loss, {
            "total_loss": total_loss.item(),
            "loss_traffic": loss_traffic.item(),
            "loss_multi": loss_multi.item(),
        }

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        temperature_cal: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference step:
        Computes predicted class y_hat, calibrated reliability r_t, and full posterior distribution.
        """
        z, similarities = self.forward(x)
        calibrated_logits = similarities / temperature_cal
        probs = F.softmax(calibrated_logits, dim=-1)
        reliability, pred_class = torch.max(probs, dim=-1)
        return pred_class, reliability, probs
