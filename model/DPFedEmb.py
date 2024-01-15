from torch import nn, optim
import utils.layer_utils as layer
from config import config_dict
from model.ResNet4Emb import ResNet18Client, ResNet18BackboneWithClassifier
import pretrain4FedEmb
import utils.test_utils as tst
import random
from utils.layer_utils import init_client_head
from torch.utils.data import ConcatDataset, DataLoader


class DPFedEmb(nn.Module):
    def __init__(self, num_clients, pretrain_loader, total_client_loaders, test_loader, test_dir, device):
        super(DPFedEmb, self).__init__()
        self.device = device
        self.epochs = config_dict['epochs']
        self.lr_alpha = config_dict['lr_fedemb_backbone']
        self.lr_beta1 = config_dict['lr_fedemb_client_backbone']
        self.lr_beta2 = config_dict['lr_fedemb_client_head']
        self.batch_size = config_dict['batch_size']
        self.total_client_loaders = total_client_loaders
        self.num_clients = num_clients
        self.pretrain_loader = pretrain_loader
        self.global_model = ResNet18BackboneWithClassifier(num_classes=10).to(device)
        self.client_models = [ResNet18Client().to(device) for i in range(num_clients)]
        self.test_loader = test_loader
        self.test_dir = test_dir

    def client_update(self, model, data_loaders):
        # backbone features
        optimizer_beta1 = optim.SGD(model.features.parameters(), lr=self.lr_beta1, momentum=0.9)
        # global_avg_pool & embedding & classifier
        non_feature_params = [param for name, param in model.named_parameters() if "features" not in name]
        optimizer_beta2 = optim.SGD(non_feature_params, lr=self.lr_beta2, momentum=0.9)
        model.train()
        total_loss = 0.0
        for data_loader in data_loaders:
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer_beta1.zero_grad()
                optimizer_beta2.zero_grad()

                output, _ = model(data)
                loss = nn.functional.cross_entropy(output, target)
                total_loss += loss.item()
                loss.backward()

                # dp for features
                for param in model.features.parameters():
                    if param.grad is not None:
                        param.grad = layer.grad_clipping((param.grad,))

                optimizer_beta1.step()
                optimizer_beta2.step()

        features_state_dict = {name: param.data for name, param in model.named_parameters() if 'features' in name}
        return features_state_dict, total_loss / (len(data_loader) * len(data_loaders))

    def federated_avg(self, models):
        glb_model = models[0]
        for key in glb_model.keys():
            if 'features' in key:
                for model in models[1:]:
                    glb_model[key] += model[key]
                glb_model[key] = glb_model[key] / len(models)
                noise = layer.add_noise(glb_model[key])
                glb_model[key] += noise
        return glb_model

    def train_eval(self):
        self.global_model = pretrain4FedEmb.pretrain(self.global_model, self.pretrain_loader, self.device)
        tst.eval_tsne_image(self.test_loader, self.global_model.backbone, self.device, self.test_dir, "before2")

        global_optimizer = optim.SGD(self.global_model.parameters(), lr=self.lr_alpha, momentum=0.9)

        for epoch in range(self.epochs):
            client_loaders = self.sample_client_data()
            client_updates = []
            epoch_loss = 0.0
            self.train()
            for client_model, data_loader in zip(self.client_models, client_loaders):
                client_model.features.load_state_dict(self.global_model.backbone.features.state_dict())
                client_model = init_client_head(client_model, self.device)
                update, loss = self.client_update(client_model, data_loader)
                epoch_loss += loss
                client_updates.append(update)

            aggregated_state_dict = self.federated_avg(client_updates)
            # update global model
            global_optimizer.zero_grad()
            for name, param in self.global_model.named_parameters():
                update_key = name.replace('backbone.', '')
                if 'features' in name:
                    param.grad = aggregated_state_dict[update_key] - param.data
            global_optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(client_loaders[0])}")

        tst.eval_tsne_image(self.test_loader, self.global_model.backbone, self.device, self.test_dir, "after2")

    def forward(self, x):
        x = x.to(self.device)
        return self.global_model(x)

    def sample_client_data(self, samples_per_client=20):
        client_loaders = []
        for _ in range(self.num_clients):
            sampled_loaders = random.sample(self.total_client_loaders, samples_per_client)
            client_loaders.append(sampled_loaders)
        return client_loaders
