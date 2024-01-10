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


"""
def evaluate_model(
    device, model: MLP, data_loader: data.DataLoader, labels_vectors: torch.Tensor
):
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
            
            true_predictions += int(pred_label == labels[0])
            # label = 0 - Benign
            false_positive += int(
                pred_label != labels[0] and (pred_label == 0 and labels[0] != 0)
            )
            false_negative += int(
                pred_label != labels[0] and (pred_label != 0 and labels[0] == 0)
            )
            predicitons_amount += 1

        accuracy = accuracy(true_predictions, predicitons_amount)
        precision = precision(true_predictions, false_positive)
        sensitivity = sensitivity(true_predictions, false_negative)
        f1 = f1(precision, sensitivity)


    print(f"Accuracy: {accuracy:4.2f}%")
    print(f"Precision: {precision:4.2f}")
    print(f"Sensitivity: {sensitivity:4.2f}")
    print(f"F1: {f1:4.2f}")
    """
