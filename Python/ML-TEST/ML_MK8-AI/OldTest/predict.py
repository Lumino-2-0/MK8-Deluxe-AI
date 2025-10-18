import torch
import torch.nn as nn
import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Définition du modèle (doit correspondre à train.py)
class ColorClassifier(nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 13)
        )

    def forward(self, x):
        return self.model(x)

def predict(rgb):
    model = ColorClassifier()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    rgb = [v / 255.0 for v in rgb]
    input_tensor = torch.tensor([rgb], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Utilisation : python predict.py R G B")
        sys.exit(1)

    try:
        r = int(sys.argv[1])
        g = int(sys.argv[2])
        b = int(sys.argv[3])
        result = predict([r, g, b])
        print("Couleur ({}, {}, {}) --> Position predite : {}".format(r, g, b, result))
    except ValueError:
        print("R, G et B doivent etre des entiers.")
