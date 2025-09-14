import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualPrototypeBank(nn.Module):
    def __init__(self, num_classes: int, prototypes_per_class: int, dim: int,
                 init_scale: float = 0.02, class_names=None):
        super().__init__()
        self.num_classes = num_classes
        self.K = prototypes_per_class
        self.dim = dim
        self.class_names = class_names
        # [C, K, D]
        self.prototypes = nn.Parameter(torch.randn(num_classes, prototypes_per_class, dim) * init_scale)

    @property
    def flat(self) -> torch.Tensor:
        # [C*K, D], L2-normalized
        p = self.prototypes
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
        return p.view(-1, self.dim)

    def expand_for_batch(self, B: int) -> torch.Tensor:
        # [B, C*K, D]
        P = self.flat
        return P.unsqueeze(0).expand(B, *P.shape)

    def aggregate_logits(self, proto_logits: torch.Tensor, reduce: str = "logsumexp") -> torch.Tensor:
        """
        proto_logits: [B, Q, C*K] from ClassPredictionHead
        returns class_logits: [B, Q, C]
        """
        B, Q, PK = proto_logits.shape
        C, K = self.num_classes, self.K
        L = proto_logits.view(B, Q, C, K)
        if reduce == "max":
            class_logits, _ = L.max(dim=-1)
        else:
            class_logits = torch.logsumexp(L, dim=-1)
        return class_logits
