import argparse
import subprocess
import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    parser = argparse.ArgumentParser(description="Projet IA MK8 - Classifieur de couleurs")
    parser.add_argument("--train", action="store_true", help="Entrainer le modele")
    parser.add_argument("--predict", nargs=3, type=int, metavar=("R", "G", "B"), help="Predire la position selon une couleur RGB")
    
    args = parser.parse_args()

    if args.train:
        subprocess.run([sys.executable, "train.py"])
    elif args.predict:
        r, g, b = args.predict
        subprocess.run([sys.executable, "predict.py", str(r), str(g), str(b)])
    else:
        print("Utilise --train pour entrainer ou --predict R G B pour predire.")

if __name__ == "__main__":
    main()
