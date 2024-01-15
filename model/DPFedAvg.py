from torch import nn, optim
import utils.layer_utils as layer
from config import config_dict
import utils.test_utils as tst


class DPFedAvg(nn.Module):
    def __init__(self, client_loaders, test_loader, device):
        super(DPFedAvg, self).__init__()
        self.device = device
        self.epochs = config_dict['epochs']
        self.lr = config_dict['lr_fedavg']
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.global_model = layer.adjust_resnet18_for_cifar10().to(device)
        self.client_models = [layer.adjust_resnet18_for_cifar10().to(device) for _ in client_loaders]

    def client_update(self, model, data_loader):
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        model.train()
        total_loss = 0.0
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            total_loss += loss
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = layer.grad_dp((param.grad,))
            optimizer.step()
        return model.state_dict(), total_loss.item()

    def federated_avg(self, models):
        glb_model = models[0]
        for key in glb_model.keys():
            for model in models[1:]:
                glb_model[key] += model[key]
            glb_model[key] = glb_model[key] / len(models)
        return glb_model

    def train_eval(self):
        epoch_accuracies = []
        for epoch in range(self.epochs):
            client_updates = []
            epoch_loss = 0.0
            self.train()
            for client_model, data_loader in zip(self.client_models, self.client_loaders):
                client_model.load_state_dict(self.global_model.state_dict())
                update, loss = self.client_update(client_model, data_loader)
                epoch_loss += loss
                client_updates.append(update)

            self.global_model.load_state_dict(self.federated_avg(client_updates))
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(self.client_loaders[0])}")

            # test accuracy
            accuracy = tst.test_model(self, self.test_loader, self.device, num_samples=1000)
            epoch_accuracies.append(accuracy)
            print(f"Epoch {epoch + 1}: Test Accuracy = {accuracy}")

        return epoch_accuracies

    def forward(self, x):
        x = x.to(self.device)
        return self.global_model(x)
