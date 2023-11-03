import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.GZSL.utils import map_layer_init
from src.models.generic import GenericModel


class NetNet(GenericModel):
    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        words_embeddings_dim: int,
        output_dim: int,
        labels_vectors: torch.Tensor,
    ):
        super().__init__(device)
        self.linear1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, words_embeddings_dim)
        self.bn3 = nn.BatchNorm1d(words_embeddings_dim)
        self.linear4 = nn.Linear(words_embeddings_dim, output_dim)
        self.ReLU = nn.LeakyReLU()

        # weight initialize
        self.linear4.weight.data = map_layer_init(w2c_vectors=labels_vectors)
        # freeze layer weights
        self.linear4.weight.requires_grad = False

        self.prev_x = torch.empty([])

    def forward(self, x):
        x = self.ReLU(self.linear1(x))
        x = self.ReLU(self.linear2(x))
        x = self.ReLU(self.linear3(x))
        x = self.linear4(x)
        x = F.softmax(x, dim=1)
        return x
