import os
import time
from datetime import datetime
from PIL import ImageGrab
import csv
from utils import ensure_dir_exists, XboxController

# === Paramètres de détection (plein écran 320x180) ===
RECT_START_POS = (144, 59)
RECT_START_SIZE = (31, 37)
RECT_END_POS = (66, 57)
RECT_END_SIZE = (188, 40)

DEBUT_RGB = (255, 0, 255)
FIN_RGB = (0, 255, 255)
COLOR_TOLERANCE = 10

SAVE_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Screens_AS'))

def is_color_match(image, top_left, size, target_color, tolerance=10):
    x, y = top_left
    w, h = size
    region = image.crop((x, y, x + w, y + h)).resize((1, 1))
    avg_color = region.getpixel((0, 0))
    return all(abs(avg_color[i] - target_color[i]) <= tolerance for i in range(3))

def main():
    session_name = input("Nom de la session : ").strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    img_folder = os.path.join(SAVE_BASE_PATH, session_name)
    ensure_dir_exists(img_folder)

    csv_path = os.path.join(SAVE_BASE_PATH, f"{session_name}.csv")
    csv_file = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file, delimiter=';')

    controller = XboxController()

    print("Attente du rectangle magenta (début de la course)...")
    while True:
        img = ImageGrab.grab().resize((320, 180))
        if is_color_match(img, RECT_START_POS, RECT_START_SIZE, DEBUT_RGB, COLOR_TOLERANCE):
            print("Détection du départ ! Début de la capture.")
            break
        time.sleep(0.1)

    # Avant la boucle principale
    frame_counter = 1  # Initialisation du compteur de frames

    try:
        while True:
            start_time = time.time()
            img = ImageGrab.grab().resize((320, 180))
            if is_color_match(img, RECT_END_POS, RECT_END_SIZE, FIN_RGB, COLOR_TOLERANCE):
                print("Rectangle cyan détecté. Fin de la capture.")
                break

            # Utilisation du compteur pour le nom de fichier
            filename = f"{str(frame_counter).zfill(5)}.png"  # Format 00001.png
            filepath = os.path.join(img_folder, filename)
            relative_path = os.path.join(session_name, filename)
            img.save(filepath)

            # Lecture des inputs du contrôleur
            x, y, a, b, xbtn, rb, lb = controller.read()
            writer.writerow([relative_path, round(x, 3), round(y, 3), a, b, xbtn, rb, lb])

            # Incrémenter le compteur pour la prochaine image
            frame_counter += 1

            # Calcul du temps écoulé et du délai pour la capture à 15 FPS
            elapsed = time.time() - start_time
            delay = max(0, (1/15) - elapsed)
            time.sleep(delay)


    except KeyboardInterrupt:
        print("Capture interrompue par l'utilisateur.")

    finally:
        csv_file.close()
        print(f"Session terminée. Données sauvegardées dans {csv_path}")

if __name__ == '__main__':
    main()