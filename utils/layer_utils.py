import torch
from config import config_dict
import torch.nn as nn
import torchvision.models as models

clip_bound_batch = config_dict['clip_bound_batch']
noise_multiplier = config_dict['noise_multiplier']
sensitivity = config_dict['sensitivity']


def adjust_resnet18_for_cifar10():
    model = models.resnet18()

    # CIFAR-10
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    return model


def init_client_head(client_model, device, embedding_dim=128, num_classes=10):
    # reset embedding layer
    client_model.embedding = nn.Linear(512, embedding_dim).to(device)
    # reset classifier layer
    if num_classes is not None:
        client_model.classifier = nn.Linear(embedding_dim, num_classes).to(device)
    return client_model


# dp hook
def grad_dp(grad_input):
    grad_wrt_input = grad_input[0]
    grad_input_shape = grad_wrt_input.size()
    batch_size = grad_input_shape[0]

    # reshape
    grad_wrt_input = grad_wrt_input.view(batch_size, -1)

    # clipping
    clip_bound = clip_bound_batch / batch_size
    grad_input_norm = torch.norm(grad_wrt_input, p=2, dim=1)
    clip_coef = clip_bound / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_input = clip_coef * grad_wrt_input

    # add noise
    noise = clip_bound * noise_multiplier * sensitivity * torch.randn_like(grad_wrt_input)
    grad_wrt_input = grad_wrt_input + noise

    # reshape to original grad shape
    grad_in_new = [grad_wrt_input.view(grad_input_shape)]
    for i in range(1, len(grad_input)):
        grad_in_new.append(grad_input[i])

    return grad_wrt_input.view(grad_input_shape)


def grad_clipping(grad_input):
    grad_wrt_input = grad_input[0]
    grad_input_shape = grad_wrt_input.size()
    batch_size = grad_input_shape[0]

    # Reshape
    grad_wrt_input = grad_wrt_input.view(batch_size, -1)

    # Clipping
    clip_bound = clip_bound_batch / batch_size
    grad_input_norm = torch.norm(grad_wrt_input, p=2, dim=1)
    clip_coef = clip_bound / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_input = clip_coef * grad_wrt_input

    return grad_wrt_input.view(grad_input_shape)


def add_noise(glb_model_val):
    return torch.randn_like(glb_model_val) * noise_multiplier * sensitivity
