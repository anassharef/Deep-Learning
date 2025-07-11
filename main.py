
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from utils import plot_tsne

import os
import sys
import time
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms, datasets

NUM_CLASSES = 10
epochs_classifier =25
epochs_autoencoder = 25


def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether to train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    parser.add_argument('--contrastive', action='store_true', default=False,
                        help='Run the contrastive learning pipeline')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Run evaluation (linear classifier training and t-SNE) using checkpoint')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint for evaluation')


    return parser.parse_args()


#################################################################################################################################################################################################################
###################################################here will implement 1.2.1 ####################################################################################################################################
#################################################################################################################################################################################################################
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, mnist=False):
        super(Autoencoder, self).__init__()
        self.mnist = mnist
        input_channels = 1 if mnist else 3

        # Encoder with BatchNorm for stability
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # 16x16 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # 8x8 -> 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )

        # Decoder with BatchNorm as well
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 8x8 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        # A deeper classifier with BatchNorm and Dropout for better regularization
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


def train_autoencoder(model, train_loader, val_loader, args):
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    alpha = 0.5  # You can tune this parameter


    for epoch in range(epochs_autoencoder):
        model.train()
        train_loss = 0.0

        for data, _ in tqdm(train_loader, desc=f"Autoencoder Epoch {epoch + 1}"):
            data = data.to(args.device)
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss_mse = mse_criterion(reconstructed, data)
            loss_mae = mae_criterion(reconstructed, data)
            loss = alpha * loss_mse + (1 - alpha) * loss_mae
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(args.device)
                reconstructed, _ = model(data)
                loss_mse = mse_criterion(reconstructed, data)
                loss_mae = mae_criterion(reconstructed, data)
                loss = alpha * loss_mse + (1 - alpha) * loss_mae
                val_loss += loss.item() * data.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs_autoencoder} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} -mae Loss: {loss_mae:.4f}, mse Loss: {loss_mse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_{"mnist" if get_args().mnist else "cifar10"}_autoencoder.pth')

    print("Autoencoder training complete!")

    model.load_state_dict(torch.load(f'best_{"mnist" if get_args().mnist else "cifar10"}_autoencoder.pth'))
    return model

def train_classifier(encoder, classifier, train_loader, val_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    best_val_acc = 0.0

    # Freeze encoder weights
    for param in encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs_classifier):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for data, labels in tqdm(train_loader, desc=f"Classifier Epoch {epoch + 1}"):
            data, labels = data.to(args.device), labels.to(args.device)
            optimizer.zero_grad()

            with torch.no_grad():
                latent = encoder(data)

            outputs = classifier(latent)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = 100. * correct / total

        # Validation
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(args.device), labels.to(args.device)
                latent = encoder(data)
                outputs = classifier(latent)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs_classifier} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), f'best_{"mnist" if get_args().mnist else "cifar10"}classifier.pth')

    print("Classifier training complete!")
    classifier.load_state_dict(torch.load(f'best_{"mnist" if get_args().mnist else "cifar10"}classifier.pth'))
    return classifier


def get_transform(mnist=False):
    if mnist:
        return transforms.Compose([
            transforms.Resize((32, 32)),  # Resize MNIST images to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])





def visualize_reconstructions(autoencoder, test_loader, args, num_images=5, save_path='reconstructions.png'):
    autoencoder.eval()
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images[:num_images].to(args.device)

    with torch.no_grad():
        reconstructed, _ = autoencoder(images)

    # Denormalize: x = x*0.5 + 0.5
    images = images.cpu() * 0.5 + 0.5
    reconstructed = reconstructed.cpu() * 0.5 + 0.5

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        if args.mnist:
            axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        else:
            axes[0, i].imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        axes[0, i].axis('off')

        if args.mnist:
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        else:
            axes[1, i].imshow(np.transpose(reconstructed[i].numpy(), (1, 2, 0)))
        axes[1, i].axis('off')
    plt.suptitle("Top: Original images | Bottom: Reconstructed images")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Reconstruction visualization saved to {save_path}")


def perform_interpolation(autoencoder, test_loader, args, idx_pair=(0, 1), num_steps=10, save_path='interpolation.png'):
    autoencoder.eval()
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    if images.shape[0] < max(idx_pair) + 1:
        raise ValueError("Not enough images in the batch for interpolation")

    img1 = images[idx_pair[0]].unsqueeze(0).to(args.device)
    img2 = images[idx_pair[1]].unsqueeze(0).to(args.device)

    with torch.no_grad():
        _, latent1 = autoencoder(img1)
        _, latent2 = autoencoder(img2)

    alphas = np.linspace(0, 1, num_steps)
    interpolated_latents = [latent1 * (1 - alpha) + latent2 * alpha for alpha in alphas]
    interpolated_latents = torch.cat(interpolated_latents, dim=0)

    with torch.no_grad():
        decoded_images = autoencoder.decoder(interpolated_latents)

    decoded_images = decoded_images.cpu() * 0.5 + 0.5

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
    for i in range(num_steps):
        if args.mnist:
            axes[i].imshow(decoded_images[i].squeeze(), cmap='gray')
        else:
            axes[i].imshow(np.transpose(decoded_images[i].numpy(), (1, 2, 0)))
        axes[i].axis('off')
    plt.suptitle("Linear Interpolation between two images")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Interpolation visualization saved to {save_path}")
    if args.mnist:
        print(
            "Analysis: The interpolated images should ideally resemble a smooth transition between two handwritten digits if the latent space is well-structured. If the digits do not appear smooth or coherent, it might indicate that the model's latent space has not fully captured the underlying structure."
        )
    else:
        print(
            "Analysis: The interpolated images should ideally exhibit a smooth transition between two CIFAR-10 images, reflecting gradual changes in object shape, color, and texture if the latent space is well-structured. Abrupt transitions or a lack of coherent blending may suggest that the model's latent space is not fully capturing the complex visual features inherent to CIFAR-10 data."
        )


def q1_2_1():
    args = get_args()
    freeze_seeds(args.seed)
    transform = get_transform(args.mnist)

    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    # Split train into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize models
    autoencoder = Autoencoder(latent_dim=args.latent_dim, mnist=args.mnist).to(args.device)

    if args.self_supervised:
        # Train autoencoder
        print("Training autoencoder...")
        autoencoder = train_autoencoder(autoencoder, train_loader, val_loader, args)

        # Visualize reconstructions for 5 random images
        visualize_reconstructions(autoencoder, test_loader, args, num_images=5, save_path='reconstructions.png')

        # Perform linear interpolation on a pair of images
        if args.mnist:
            perform_interpolation(autoencoder, test_loader, args, idx_pair=(0, 1), num_steps=10,
                                  save_path='interpolation_mnist.png')
        else:
            perform_interpolation(autoencoder, test_loader, args, idx_pair=(0, 1), num_steps=10,
                                  save_path='interpolation_cifar10.png')


        # Plot TSNE visualizations for latent and image space
        print("Plotting TSNE visualizations...")
        plot_tsne(autoencoder.encoder, test_loader, args.device)

        # Rename the TSNE output files based on dataset type
        if args.mnist:
            os.rename("latent_tsne.png", "mnist1.2.1_latent_tsne.png")
            os.rename("image_tsne.png", "mnist1.2.1_image_tsne.png")
        else:
            os.rename("latent_tsne.png", "cifar1.2.1_latent_tsne.png")
            os.rename("image_tsne.png", "cifar1.2.1_image_tsne.png")
#
        # Initialize and train classifier
        classifier = Classifier(latent_dim=args.latent_dim, num_classes=NUM_CLASSES).to(args.device)
        print("Training classifier...")
        classifier = train_classifier(autoencoder.encoder, classifier, train_loader, val_loader, args)

        # Evaluate on test set
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(args.device), labels.to(args.device)
                latent = autoencoder.encoder(data)
                outputs = classifier(latent)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        test_acc = 100. * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")
    else:
        # Joint training (not implemented in this example)
        pass

#################################################################################################################################################################################################################
###################################################here will implement 1.2.2 ####################################################################################################################################
#################################################################################################################################################################################################################
# A simple wrapper that flattens images before feeding them to the encoder
# A more complex CNN encoder architecture for improved accuracy
class EncoderCNN(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(EncoderCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduces spatial dims by 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # global pooling
        )
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def q1_2_2(args):
    # Setup dataset-specific transform and input dimensions/channels
   
    freeze_seeds(args.seed)


    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        input_channels = 1
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        input_channels = 3

    # Load datasets
    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Build the CNN encoder and classifier models
    encoder = EncoderCNN(input_channels, args.latent_dim).to(args.device)
    classifier = nn.Linear(args.latent_dim, NUM_CLASSES).to(args.device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        encoder.train()
        classifier.train()
        running_loss = 0.0
        running_mae = 0.0

        # Progress bar for training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for data, target in pbar:
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()

            latent = encoder(data)
            output = classifier(latent)

            # Cross entropy loss for classification
            loss = criterion(output, target)

            # Compute MAE loss between softmax probabilities and one-hot encoded targets
            softmax_out = torch.nn.functional.softmax(output, dim=1)
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).float()
            mae_loss = torch.nn.functional.l1_loss(softmax_out, target_one_hot)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mae += mae_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae_loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        avg_mae = running_mae / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}")

        # Evaluation on test set
        encoder.eval()
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(args.device), target.to(args.device)
                latent = encoder(data)
                output = classifier(latent)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        accuracy = 100.0 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot TSNE visualizations using the encoder directly (it expects images in original shape)
    plot_tsne(encoder, test_loader, args.device)

    # Rename the generated plot files to include dataset and section info
    if args.mnist:
        os.rename("latent_tsne.png", "mnist1.2.2_latent_tsne.png")
        os.rename("image_tsne.png", "mnist1.2.2_image_tsne.png")
    else:
        os.rename("latent_tsne.png", "cifar1.2.2_latent_tsne.png")
        os.rename("image_tsne.png", "cifar1.2.2_image_tsne.png")

    print("Experiment complete. TSNE plots saved and renamed accordingly.")




#################################################################################################################################################################################################################
###################################################here will implement 1.2.3 ####################################################################################################################################
#################################################################################################################################################################################################################
# ----- Dataset Wrappers and DataLoader Functions -----
# ----- Dataset Wrappers and DataLoader Functions -----
def repeat_channel(x):
    # Converts a 1-channel image tensor to 3 channels.
    return x.repeat(3, 1, 1)



# -------------------------
# Network definitions (from resnet_big.py)
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        return (out, preact) if self.is_last else out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        return (out, preact) if self.is_last else out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# We assume here that we're always using resnet50.
model_dict = {
    'resnet50': [resnet50, 2048],
}


class SupConResNet(nn.Module):
    """Backbone with projection head for Supervised Contrastive Learning"""

    def __init__(self, name='resnet50', head='mlp', feat_dim=128, in_channel=3):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(in_channel=in_channel)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


# -------------------------
# Loss function (from losses.py)
# -------------------------
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.
    When labels is None, it degenerates to SimCLR unsupervised loss."""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device('cuda' if features.is_cuda else 'cpu')
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


# -------------------------
# Utility functions (from util.py)
# -------------------------
class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, optimizer, epoch, epochs):
    # Simple cosine annealing schedule
    eta_min = lr * 0.001
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(warmup_from, warmup_to, optimizer, epoch, batch_id, total_batches, warm_epochs):
    if epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(model, learning_rate, momentum=0.9, weight_decay=1e-4):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def save_model(model, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, save_file)
    del state


# -------------------------
# Data Loader and Model Setup for Pretraining
# -------------------------
def set_loader(args, image_size=32):
    if args.mnist:
        # MNIST: convert grayscale to 3 channels.
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channel),
            normalize,
        ])
        dataset = datasets.MNIST(root=args.data_path, train=True,
                                 transform=TwoCropTransform(transform), download=True)
    else:
        # CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.CIFAR10(root=args.data_path, train=True,
                                   transform=TwoCropTransform(transform), download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=4, pin_memory=True)
    return loader


def set_model(args):
    # Use 3 channels always. For pretraining, we build SupConResNet.
    model = SupConResNet(name='resnet50', feat_dim=args.latent_dim, in_channel=3)
    criterion = SupConLoss(temperature=0.07)
    model = model.to(args.device)
    criterion = criterion.to(args.device)
    if torch.cuda.device_count() > 1 and args.device == 'cuda':
        model.encoder = nn.DataParallel(model.encoder)
    cudnn.benchmark = True
    return model, criterion


# -------------------------
# Pretraining (Contrastive) Training Loop
# -------------------------
def pretrain(args):
    train_loader = set_loader(args, image_size=32)
    model, criterion = set_model(args)
    learning_rate = 0.05
    optimizer = set_optimizer(model, learning_rate)
    epochs = 10  # Change as needed
    for epoch in range(1, epochs + 1):
        adjust_learning_rate(learning_rate, optimizer, epoch, epochs)
        loss = train_contrastive(train_loader, model, criterion, optimizer, args.device, epoch, epochs)
        print("Epoch {} completed. Average Loss: {:.4f}".format(epoch, loss))
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_file = os.path.join("checkpoints", "ckpt_epoch_{}.pth".format(epoch))
            save_model(model, optimizer, epoch, save_file)
    os.makedirs("checkpoints", exist_ok=True)
    save_file = os.path.join("checkpoints", "last.pth")
    save_model(model, optimizer, epochs, save_file)


def train_contrastive(train_loader, model, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    # Warmup settings (optional)
    warmup_from = 0.01
    warmup_to = 0.05
    warm_epochs = 5
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]
        warmup_learning_rate(warmup_from, warmup_to, optimizer, epoch, idx, len(train_loader), warm_epochs)
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                  .format(epoch, total_epochs, idx + 1, len(train_loader), losses.avg))
    return losses.avg


# -------------------------
# Linear Classifier for Evaluation
# -------------------------
class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def evaluate(args):
    if args.checkpoint is None:
        print("Please provide a checkpoint path using --checkpoint for evaluation.")
        sys.exit(0)
    # Build the same SupConResNet and load checkpoint
    model, _ = set_model(args)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded from", args.checkpoint)
    # Freeze encoder and remove the projection head for evaluation.
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Create a linear classifier.
    num_classes = 10  # Both MNIST and CIFAR10 have 10 classes.
    feat_dim = model_dict['resnet50'][1]  # Typically 2048 for resnet50.
    classifier = LinearClassifier(feat_dim, num_classes).to(args.device)

    # Prepare datasets (without two-crop transforms)
    if args.mnist:
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channel),
            normalize,
        ])
        train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transform, download=True)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.SGD(classifier.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)
    epochs = 5  # Adjust as needed

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0
        total_correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            # Extract features using the frozen encoder.
            with torch.no_grad():
                feats = model.encoder(images)
            outputs = classifier(feats)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()
        train_loss = total_loss / total
        train_acc = 100*total_correct / total
        print("Epoch {}: Train Loss: {:.4f}, Train Acc: {:.4f}".format(epoch, train_loss, train_acc))

    # Evaluation on test set.
    classifier.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            feats = model.encoder(images)
            outputs = classifier(feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())
    test_loss = total_loss / total
    test_acc = 100*total_correct / total
    print("Test Loss: {:.4f}, Test Acc: {:.4f}".format(test_loss, test_acc))

    # t-SNE visualization.
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        all_features = torch.cat(all_features, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_features)
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("t-SNE of Learned Representations")
        plt.savefig("tsne.png")
        plt.show()
    except ImportError:
        print("Please install scikit-learn and matplotlib for t-SNE visualization.")


# -------------------------
# Main Function
# -------------------------
def q1_2_3():
    args = get_args()
    # Set seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    if args.eval:
        evaluate(args)
    else:
        if not args.contrastive:
            print("This script is set up for contrastive pretraining (--contrastive) or evaluation (--eval). Exiting.")
            sys.exit(0)
        pretrain(args)
if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)


    if args.contrastive:
        q1_2_3()
    else:
        if args.self_supervised:
            q1_2_1()
        else:
            q1_2_2(args)

