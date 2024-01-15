import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
# from autodp import rdp_acct, rdp_bank
import warnings
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import numpy as np
from config import config_dict

# test accuracy
def test_model(model, test_loader, device, num_samples=1000):
    model.eval()
    test_dataset = test_loader.dataset

    num_test_samples = len(test_dataset)
    indices = torch.randperm(num_test_samples)[:num_samples]

    random_test_loader = DataLoader(test_dataset, sampler=SubsetRandomSampler(indices),
                                    batch_size=test_loader.batch_size)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in random_test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def draw_accuracy(epoch_accuracies, folder_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # save img
    save_path = os.path.join(folder_path, "Accuracy Over Epochs")
    plt.savefig(save_path)


# calculate epsilon based on Renyi DP
# def cal_privacy_budget(cfg):
#     delta = cfg['delta']
#     batch_size = cfg['batch_size']
#     sigma = cfg['noise_multiplier']
#     n_steps = cfg['total_iterations']
#     func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
#
#     acct = rdp_acct.anaRDPacct()
#     acct.compose_subsampled_mechanism(func, cfg['sample_prop'], coeff=n_steps * batch_size)
#     epsilon = acct.get_eps(delta)
#     print(f"ε = {epsilon}, δ = {delta}")


# Perform t-SNE on the latent variables
# & show reconstruction image of first and last epoch
def eval_tsne_image(test_loader, backbone, device, folder_path, file_name, num_samples=3000):
    fig, ax = plt.subplots()
    output = []
    real_labels = []
    backbone.eval()
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        output.append(backbone(data))
        real_labels.append(targets)
        if len(output) >= num_samples:
            break
    output = torch.cat(output).cpu()
    real_labels = torch.cat(real_labels).cpu()
    plotdistribution(real_labels, output, ax,
                     map_color={0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'k', 5: 'm', 6: 'c', 7: 'pink', 8: 'grey',
                                9: 'blueviolet'})
    save_path = os.path.join(folder_path, f"TSNE-{file_name}")
    plt.savefig(save_path)

def plotdistribution(Label, Mat, ax, map_color):
    warnings.filterwarnings('ignore', category=FutureWarning)
    tsne = TSNE(n_components=2, random_state=0)
    if Mat.requires_grad:
        Mat = Mat.detach()
    Mat = Mat.numpy()
    Mat = Mat.reshape(Mat.shape[0], -1)
    Mat = tsne.fit_transform(Mat[:])

    x = Mat[:, 0]
    y = Mat[:, 1]

    Label = [label.item() for label in Label]
    color = [map_color[label] for label in Label]

    ax.scatter(np.array(x), np.array(y), s=5, c=color, marker='o')  # The scatter function only supports array type data

    # add labels
    legend_elements = []
    for label, color in map_color.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5, label=label))
    ax.legend(handles=legend_elements, title='Label', loc='upper right', handlelength=0.8, handleheight=0.8)