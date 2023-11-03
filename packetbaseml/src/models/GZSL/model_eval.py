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
    true_predictions, predicitons_amount = 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_vectors = labels_vectors.to(device)

            pred_input = model(inputs)
            pred_label, _ = find_closest_vector(
                vector=pred_input[0], labels_vectors=labels_vectors
            )
            true_predictions += int(pred_label == labels[0])
            predicitons_amount += 1

        accuracy = 100.0 * true_predictions / predicitons_amount

    print(f"Accuracy of the model: {accuracy:4.2f}%")
