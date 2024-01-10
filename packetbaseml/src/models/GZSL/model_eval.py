import torch
from src.models.GZSL.model import NetNet
import torch.utils.data as data


def find_closest_vector(vector: torch.Tensor, labels_vectors: torch.Tensor) -> float:
    min_dist = float("inf")
    min_dist_label = float("inf")
    pdist = torch.nn.PairwiseDistance(p=2)
    for label, label_vector in enumerate(labels_vectors, start=0):
        dist = pdist(label_vector, vector)
        if dist < min_dist:
            min_dist = dist
            min_dist_label = float(label)
    return min_dist_label


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

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_vectors = labels_vectors.to(device)

            pred_input = model(inputs)
            pred_label = find_closest_vector(
                vector=pred_input[0], labels_vectors=labels_vectors
            )
            true_predictions += int(pred_label == labels[0])
            # label = 0 - Benign
            false_positive += int(
                pred_label != labels[0] and (pred_label == 0 and labels[0] != 0)
            )
            false_negative += int(
                pred_label != labels[0] and (pred_label != 0 and labels[0] == 0)
            )
            predicitons_amount += 1

        accuracy = 100.0 * true_predictions / predicitons_amount
        precision = 100.0 * true_predictions / (true_predictions + false_positive)
        sensitivity = 100.0 * true_predictions / (true_predictions + false_negative)
        f1 = 2 * precision * sensitivity / (precision + sensitivity)

    print(f"Accuracy: {accuracy:4.2f}%")
    print(f"Precision: {precision:4.2f}")
    print(f"Sensitivity: {sensitivity:4.2f}")
    print(f"F1: {f1:4.2f}")
