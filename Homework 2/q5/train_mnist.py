"""Train the MNIST network for HW2 Problem 5 and export it to ONNX.

The network takes a normalized image of shape (1, 1, 28, 28):
    x_normalized = (x - 0.1307) / 0.3081
ConstraintFlow applies this normalization before the network, so the
network itself has no normalization layer.
"""
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms

MEAN, STD = 0.1307, 0.3081


def make_model(hidden=100, depth=3):
    layers = [nn.Flatten()]
    width = 28 * 28
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.ReLU()]
        width = hidden
    layers += [nn.Linear(width, 10)]
    return nn.Sequential(*layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="mnist_relu_3_100.onnx")
    args = parser.parse_args()

    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train = datasets.MNIST(args.data, train=True, download=True, transform=tf)
    test = datasets.MNIST(args.data, train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000)

    model = make_model(args.hidden, args.depth)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(x).argmax(1) == y).sum().item() for x, y in test_loader)
        print(f"epoch {epoch + 1}: test accuracy {correct / len(test):.4f}")

    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), args.out,
                      input_names=["input"], output_names=["output"], dynamo=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
