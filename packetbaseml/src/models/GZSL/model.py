import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from src.models.GZSL.utils import map_layer_init


class NetNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        words_embeddings_dim: int,
        output_dim: int,
        labels_vectors: torch.Tensor,
    ):
        super().__init__()
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


def print_model_layer_gradients(model: NetNet) -> None:
    for name, layer in model.named_modules():
        if len(list(layer.named_modules())) == 1 and name != "ReLU":
            print(f"Layer: {name}\nGradients: {layer.weight.grad}")


def train_model(
    device,
    model: NetNet,
    epochs: int,
    data_loader: data.DataLoader,
    loss_fn: nn.MSELoss,
    additional_eps=1e-06,
):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            if not torch.isfinite(inputs).all():
                inputs = torch.nan_to_num(inputs)  # removing nan values
            if additional_eps:
                inputs = torch.add(inputs, additional_eps)

            labels = labels.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels.to(torch.long))
            loss.backward(retain_graph=True)
            model.optim.step()
            model.optim.zero_grad()

        print(f"Epoch: {epoch}, loss: {loss.item():.3}")


def find_closest_vector(vector: torch.Tensor, labels_vectors: torch.Tensor) -> tuple:
    min_dist = float("inf")
    min_dist_label = float("inf")
    pdist = torch.nn.PairwiseDistance(p=2)
    for label, label_vector in enumerate(labels_vectors, start=0):
        dist = pdist(label_vector, vector)
        if dist < min_dist:
            min_dist = dist
            min_dist_label = float(label)
    return min_dist_label, min_dist


def evaluate_model(
    device, model: NetNet, data_loader: data.DataLoader, labels_vectors: torch.Tensor
):
    model.eval()
    true_predictions, predicitons_amount = 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_vectors = labels_vectors.to(device)

            pred_input = model(inputs)
            pred_label, dist = find_closest_vector(
                vector=pred_input[0], labels_vectors=labels_vectors
            )
            true_predictions += int(pred_label == labels[0])
            predicitons_amount += 1

        accuracy = 100.0 * true_predictions / predicitons_amount

    print(f"Accuracy of the model: {accuracy:4.2f}%")
