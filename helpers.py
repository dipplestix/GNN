import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.datasets as tg_datasets
from torch_geometric.transforms import NormalizeFeatures


def evaluate(model, data):
    X = data.x
    edge_index = data.edge_index
    labels = data.y
    test_mask = data.test_mask
    model.eval()
    output = model(X, edge_index)
    with torch.no_grad():
        _, pred = torch.max(output[test_mask], dim=1)
        correct = (pred == labels[test_mask]).sum().item()
        accuracy = correct / test_mask.sum().item()
        print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def train(data, model, num_epochs):
    X = data.x
    labels = data.y
    edge_index = data.edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X, edge_index)

        # Compute the loss only for the labeled nodes
        loss = criterion(output[train_mask], labels[train_mask])

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            _, pred = torch.max(output[val_mask], dim=1)
            correct = (pred == labels[val_mask]).sum().item()
            accuracy = correct / val_mask.sum().item()
            if epoch % 100 == 99:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation accuracy: {accuracy:.4f}")
    return model


def load_cora(device):
    # Load the Cora dataset
    cora_dataset = tg_datasets.Planetoid(root='.', name='Cora', transform=NormalizeFeatures())

    # Get the data object
    data = cora_dataset[0]

    data = data.to(device)

    return data


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
