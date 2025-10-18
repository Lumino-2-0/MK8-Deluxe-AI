import torch
if torch.cuda.is_available():
    print("Le GPU est utilise")
    device = torch.device("cuda")  # Utilise le GPU
else:
    print("Le CPU est utilise")
    device = torch.device("cpu")  # Utilise le CPU
