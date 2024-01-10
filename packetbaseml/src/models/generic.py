import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


class GenericModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def print_model_layer_gradients(self) -> None:
        for name, layer in self.named_modules():
            if len(list(layer.named_modules())) == 1 and name != "ReLU":
                print(f"Layer: {name}\nGradients: {layer.weight.grad}")

    def train_model(
        self,
        epochs: int,
        loss_fn,
        data_loader: data,
        additional_eps=1e-06,
    ) -> None:
        self.train()
        for epoch in range(epochs):
            for inputs, labels in data_loader:
                if not torch.isfinite(inputs).all():
                    inputs = torch.nan_to_num(inputs)  # removing nan values
                if additional_eps:
                    inputs = torch.add(inputs, additional_eps)

                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                outputs = self(inputs)
                loss = loss_fn(outputs, labels.to(torch.float))
                loss.backward(retain_graph=True)
                self.optim.step()
                self.optim.zero_grad()

            print(f"Epoch: {epoch}, loss: {loss.item():.3}")
