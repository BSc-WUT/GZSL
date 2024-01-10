import torch
import torch.utils.data as data

from src.models.operations import accuracy, sensitivity, precision, f1
from src.models.GZSL.model import NetNet


def find_closest_vector(
    vector: torch.Tensor, labels_vectors: torch.Tensor
) -> torch.Tensor:
    min_dist = float("inf")
    min_dist_label = -1
    pdist = torch.nn.PairwiseDistance(p=2)
    for label, label_vector in enumerate(labels_vectors, start=0):
        dist = pdist(
            label_vector.unsqueeze(0).to(vector.device), vector.unsqueeze(0)
        ).item()
        if dist < min_dist:
            min_dist = dist
            min_dist_label = label
    return torch.tensor([min_dist_label], dtype=torch.float32, device=vector.device)


def evaluate_model(
    device, model: NetNet, data_loader: data.DataLoader, labels_vectors: torch.Tensor
):
    model.eval()
    true_predictions, predicitons_amount, false_negative, false_positive = (
        0.0,
        0.0,
        0.0,
        0.0,
    )

    zero_tensor = torch.tensor(0, device=device)  # Zero tensor on the same device

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_vectors = labels_vectors.to(device)

            pred_inputs = model(inputs)
            pred_labels = torch.cat(
                [
                    find_closest_vector(
                        vector=pred_input, labels_vectors=labels_vectors
                    )
                    for pred_input in pred_inputs
                ]
            ).to(device)

            true_predictions += (pred_labels == labels).sum().item()
            false_positive += (
                (
                    (pred_labels != labels)
                    & (pred_labels == zero_tensor)
                    & (labels != zero_tensor)
                )
                .sum()
                .item()
            )
            false_negative += (
                (
                    (pred_labels != labels)
                    & (pred_labels != zero_tensor)
                    & (labels == zero_tensor)
                )
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
