import torch
import torch.nn as nn
from config import config_dict
import os

epochs = config_dict['pretrain_epochs']
lr = config_dict['pretrain_lr']
pretrained_model_file = config_dict['pretrained_model_file']


def pretrain(backbone, pretrain_loader, device):
    if os.path.isfile(pretrained_model_file):
        print("Loading pretrained model...")
        backbone.load_state_dict(torch.load(pretrained_model_file))
        backbone.to(device)
        return backbone

    backbone.to(device)
    backbone.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(backbone.parameters(), lr=lr, momentum=0.9)

    print('Pretraining Begins.')

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(pretrain_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = backbone(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch + 1}] loss: {running_loss / len(pretrain_loader)}")

    print('Finished Pretraining')
    torch.save(backbone.state_dict(), pretrained_model_file)
    return backbone
