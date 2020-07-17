import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LinearEmbedding"]


class LinearEmbedding(nn.Module):
    def __init__(self, base, output_size=64, embedding_size=64, normalize=True):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize

    def forward(self, x, get_ha=True):
        if get_ha:
            b1, b2, b3, pool, out = self.base(x, True)
        else:
            pool = self.base(x)

        pool = pool.view(x.size(0), -1)
        embedding = self.linear(pool)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        if get_ha:
            return b1, b2, b3, pool, embedding

        return embedding
