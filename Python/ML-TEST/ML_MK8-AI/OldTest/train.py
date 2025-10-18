import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import random

class ColorClassifier(nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # On prédit un label numérique aléatoire
        )

    def forward(self, x):
        return self.model(x)

def train_model():
    # Chargement du fichier CSV
    data = pd.read_csv("colors.csv", header=None)

    # Récupère les valeurs RGB et les labels
    X = data.iloc[:, :3].values.astype(np.float32) / 255.0
    y = data.iloc[:, 3].values.astype(np.int64)

    # S'assurer que les labels sont bien randomisés
    unique_labels = list(set(y))
    random.shuffle(unique_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y])

    # Diviser en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Conversion en tenseurs PyTorch
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    # Initialisation du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorClassifier().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraînement
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoque {} : perte {:.4f}".format(epoch + 1, running_loss / len(train_loader)))

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "model.pth")
    print("Modele entraine et sauvegarde.")


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.round().squeeze().long()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print("Precision sur le jeu de test : {:.2f}".format(correct / total))

if __name__ == "__main__":
    train_model()
