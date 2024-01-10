import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from src.models.generic import GenericModel
from src.models.MLP.model import MLP
from src.models.operations import accuracy, sensitivity, precision, f1


class MLP(GenericModel):
    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__(device)
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, output_dim)
        self.ReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.ReLU(self.linear1(x))
        x = self.ReLU(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim=1)
        return x


def evaluate_model(device, model: MLP, data_loader: data.DataLoader):
    model.eval()
    true_predictions, predicitons_amount, false_negative, false_positive = (
        0.0,
        0.0,
        0.0,
        0.0,
    )

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
            pred_labels = torch.sigmoid(preds)
            pred_labels = pred_labels.to(device)

            true_predictions += (pred_labels == labels).sum().item()
            false_positive += (
                ((pred_labels != labels) & (pred_labels == 0) & (labels != 0))
                .sum()
                .item()
            )
            false_negative += (
                ((pred_labels != labels) & (pred_labels != 0) & (labels == 0))
                .sum()
                .item()
            )
            predicitons_amount += len(pred_labels)

        acc = accuracy(true_predictions, predicitons_amount)
        prec = precision(true_predictions, false_positive)
        sens = sensitivity(true_predictions, false_negative)
        f1_value = f1(precision, sensitivity)

    print(f"Accuracy: {acc:4.2f}%")
    print(f"Precision: {prec:4.2f}")
    print(f"Sensitivity: {sens:4.2f}")
    print(f"F1: {f1_value:4.2f}")
