import numpy as np
import torch


def map_layer_init(w2c_vectors: list) -> torch.Tensor:
    vectors = np.asarray(w2c_vectors, dtype=float)
    vectors = torch.from_numpy(vectors)
    return vectors[:, -1, :].to(torch.float32)
