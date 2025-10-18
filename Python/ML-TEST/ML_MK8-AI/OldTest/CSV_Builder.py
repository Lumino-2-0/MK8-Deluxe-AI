import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MK8Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "actions.csv")
        self.data = pd.read_csv(self.csv_path)

        self.transform = transforms.Compose([
            transforms.Resize((180, 320)),  # Format de tes captures
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        action = {
            "steering": float(row["steering"]),
            "throttle": float(row["throttle"]),
            "brake": float(row["brake"])
        }

        return image, action
