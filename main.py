import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, Subset
from config import config_dict
import sys
import model.Centralized as centralized
import model.DPFedAvg as dpFedAvg
import model.DPFedEmb as dpFedEmb
import os
import utils.test_utils as tst
import numpy as np

# Configure GPU or CPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper params
batch_size = config_dict['batch_size']
num_clients = config_dict['num_clients']
num_users = 500
num_classes = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def main():
    # Parsing Command Line Parameters
    if len(sys.argv) < 2:
        print("Please input the model type as argv[1]："
              "1. Centralized ResNet; 2. DP-FedAvg; 3. DP-FedEmb.")
        return

    model_param = int(sys.argv[1])
    # Create directories to save generated images, evaluation
    test_dir = 'test'

    # dataset
    if model_param == 0:
        test_dir += '_centralized'
    elif model_param == 1:
        test_dir += '_dp_fedavg'
        # partition the data to clients
        client_indices = [list(
            range(i * len(train_dataset) // num_clients, (i + 1) * len(train_dataset) // num_clients))
            for i in range(num_clients)]
        client_loaders = [DataLoader(train_dataset, batch_size=64, sampler=SubsetRandomSampler(indices)) for
                          indices in client_indices]
    elif model_param == 2:
        test_dir += '_dp_fedemb'
        # for pretrain
        pretrain_ratio = 0.3
        num_train = len(train_dataset)
        num_pretrain = int(num_train * pretrain_ratio)
        num_client_train = num_train - num_pretrain
        pretrain_dataset, client_train_dataset = random_split(train_dataset, [num_pretrain, num_client_train])
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

        num_samples_per_user = len(client_train_dataset) // num_users
        remaining_samples = len(client_train_dataset) % num_users
        client_datasets = []

        start_idx = 0
        for user in range(num_users):
            end_idx = start_idx + num_samples_per_user + (1 if user < remaining_samples else 0)
            indices = list(range(start_idx, end_idx))
            user_dataset = Subset(client_train_dataset, indices)
            client_datasets.append(user_dataset)
            start_idx = end_idx

        total_client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]
    elif model_param == 3:
        test_dir += '_dp_fedavg_heterogeneity'
        labels = train_dataset.targets

        client_indices = []
        for i in range(0, 10, 2):
            # indices for label == i or label == i+1
            indices = [idx for idx, label in enumerate(labels) if label == i or label == i + 1]
            client_indices.append(indices)
        client_loaders = [DataLoader(train_dataset, batch_size=64, sampler=SubsetRandomSampler(indices)) for
                          indices in client_indices]
    elif model_param == 4:
        test_dir += '_dp_fedemb_heterogeneity'
        # for pretrain
        pretrain_ratio = 0.3
        num_train = len(train_dataset)
        num_pretrain = int(num_train * pretrain_ratio)
        num_client_train = num_train - num_pretrain
        pretrain_dataset, client_train_dataset = random_split(train_dataset, [num_pretrain, num_client_train])
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

        # partition to users
        user_classes = {i: np.random.choice(range(num_classes), 2, replace=False) for i in range(num_users)}
        client_datasets = []
        for user in range(num_users):
            indices = [i for i in range(len(client_train_dataset)) if
                       train_dataset.targets[client_train_dataset.indices[i]] in user_classes[user]]
            user_dataset = Subset(client_train_dataset, indices)
            client_datasets.append(user_dataset)
        total_client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]

    # model
    if model_param == 0:
        model = centralized.Centralized(train_loader, test_loader, device)
    elif model_param == 1:
        model = dpFedAvg.DPFedAvg(client_loaders, test_loader, device)
    elif model_param == 2:
        model = dpFedEmb.DPFedEmb(num_clients, pretrain_loader, total_client_loaders,
                                  test_loader, test_dir, device).to(device)
    elif model_param == 3:
        model = dpFedAvg.DPFedAvg(client_loaders, test_loader, device)
    elif model_param == 4:
        model = dpFedEmb.DPFedEmb(num_clients, pretrain_loader, total_client_loaders,
                                  test_loader, test_dir, device).to(device)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if model_param == 2 or model_param == 4:
        model.train_eval()
    else:
        epoch_accuracies = model.train_eval()
        tst.draw_accuracy(epoch_accuracies, test_dir)

    # if model_param != 0:
    #     tst.cal_privacy_budget(config_dict)


if __name__ == "__main__":
    main()
