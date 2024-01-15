import torch.optim as optim
import torch.nn as nn
from config import config_dict
from utils.layer_utils import adjust_resnet18_for_cifar10
import utils.test_utils as tst


class Centralized(nn.Module):
    def __init__(self, train_loader, test_loader, device):
        super(Centralized, self).__init__()
        self.device = device
        self.epochs = config_dict['epochs']
        self.lr = config_dict['lr_centralized']
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.embedding_model = adjust_resnet18_for_cifar10().to(device)
        self.optimizer = optim.SGD(self.embedding_model.parameters(), lr=self.lr, momentum=0.9)

    def forward(self, x):
        x = x.to(self.device)
        return self.embedding_model(x)

    def train_eval(self):
        epoch_accuracies = []
        # Training loop
        for epoch in range(self.epochs):
            running_loss = 0.0
            self.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)

                loss = nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}")

            # test accuracy
            accuracy = tst.test_model(self, self.test_loader, self.device, num_samples=1000)
            epoch_accuracies.append(accuracy)
            print(f"Epoch {epoch + 1}: Test Accuracy = {accuracy}")

        return epoch_accuracies
