from __future__ import annotations

import torch

from FeatureExtractor import Embedding


class ComplexityMetric:
    input_type = "continuous"

    def __call__(self, embedding: Embedding) -> torch.Tensor:
        if not isinstance(embedding, Embedding):
            raise TypeError("ComplexityMetric input must be an Embedding")

        if self.input_type == "continuous":
            embedding = embedding.to_continuous()
        elif self.input_type == "discrete":
            embedding = embedding.to_discrete()
        else:
            raise ValueError("input_type must be 'continuous' or 'discrete'")

        if embedding.tensor.ndim != 3:
            raise ValueError("Embedding tensor must have shape (B,T,D)")
        return self.score(embedding)

    def score(self, embedding: Embedding) -> torch.Tensor:
        if embedding.tensor.shape[0] == 0:
            return torch.empty(0, device=embedding.tensor.device)
        return torch.stack([self.score_one(embedding, i) for i in range(embedding.tensor.shape[0])])

    def score_one(self, embedding: Embedding, i: int) -> torch.Tensor:
        raise NotImplementedError
