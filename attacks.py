import torch
from torch import nn
from typing import List, Callable


class GradientSignAttacker:

    def __init__(
            self,
            model: nn.Module,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            input_shape: List,
            n_classes: int
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.adversaries_labels = torch.LongTensor(list(range(n_classes)))
        adversaries_shape = [n_classes] + input_shape
        self.adversaries = torch.empty(adversaries_shape, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.adversaries)
        self.adversaries.register_hook(self.save_adversaries_grad)
        self.adversaries_grad = None

    def save_adversaries_grad(self, grad):
        self.adversaries_grad = grad

    def fit_adversaries(self, num_epochs, eta=0.01):
        for _ in range(num_epochs):
            out = self.model(self.adversaries)
            loss = self.loss_fn(out, self.adversaries_labels)
            self.model.zero_grad()
            loss.backward()
            grad_sign = torch.sign(self.adversaries_grad)
            with torch.no_grad():
                new_value = torch.clamp(self.adversaries - eta * grad_sign, 0, 1)
                self.adversaries.data.copy_(new_value)

    def attack(self, x, class_idx, alpha=0.5):
        noise = self.adversaries[class_idx]
        return torch.max(alpha * x, (1 - alpha) * noise)

    def attack_additive(self, x, class_idx, eps=0.5):
        noise = self.adversaries[class_idx]
        return torch.clamp(x + eps * noise, 0, 1)

