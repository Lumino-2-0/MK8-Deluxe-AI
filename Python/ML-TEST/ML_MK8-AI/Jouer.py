import time
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model
import pyvjoy
import ScreenHooker



FRAME_RATE = 15
FRAME_DELAY = 1 / FRAME_RATE
MODEL_PATH = 'final_model.h5'
IMG_SIZE = (64, 64)
model = load_model(MODEL_PATH)

j = pyvjoy.VJoyDevice(1)


def reset_gamepad():
    j.set_button(1, 0)
    j.set_button(2, 0)
    j.set_button(3, 0)
    j.set_button(4, 0)
    j.set_button(5, 0)
    j.reset()
    j.set_axis(pyvjoy.HID_USAGE_RX, 16384)
    j.set_axis(pyvjoy.HID_USAGE_RY, 16384)
    print("Stick centré et boutons relâchés.")
    time.sleep(1)


def grab_screen():
    return ImageGrab.grab()


def apply_action_to_gamepad(prediction):
    buttons = {
        'B': round(prediction[2]),
        'A': round(prediction[3]),
        'X': round(prediction[4]),
        'ZL': round(prediction[5]),
        'ZR': round(prediction[6])
    }

    # prediction[0] et [1] sont entre 0 et 1
    stick_x = float(prediction[0])  # 0.0 = gauche, 0.5 = centre, 1.0 = droite
    stick_y = float(prediction[1])  # 0.0 = haut,   0.5 = centre, 1.0 = bas

    axis_rx = int(stick_x * 32768)
    axis_ry = int(stick_y * 32768)

    axis_rx = max(0, min(axis_rx, 32768))
    axis_ry = max(0, min(axis_ry, 32768))

    print(f"Prediction: {prediction}")
    print(f"Action faites :")
    print(f"Stick: X={axis_rx}, Y={axis_ry}")
    print(f"Boutons: A={buttons['A']}, B={buttons['B']}, X={buttons['X']}, ZL={buttons['ZL']}, ZR={buttons['ZR']}")

    # Apply buttons
    j.set_button(1, buttons['A'])
    j.set_button(2, buttons['B'])
    j.set_button(3, buttons['X'])
    j.set_button(4, buttons['ZR'])
    j.set_button(5, buttons['ZL'])

    # Apply stick
    j.set_axis(pyvjoy.HID_USAGE_RX, axis_rx)
    j.set_axis(pyvjoy.HID_USAGE_RY, axis_ry)
    


reset_gamepad() # Remise à 0 surtout le stick Gauche !

print("Recherche du signal de départ (rectangle magenta)...")

while True:
    screen = grab_screen().resize((320, 180))
    if ScreenHooker.is_color_match(
        screen,
        ScreenHooker.RECT_START_POS,
        ScreenHooker.RECT_START_SIZE,
        ScreenHooker.DEBUT_RGB,
        ScreenHooker.COLOR_TOLERANCE
    ):
        print("Début de course détecté.")
        break
    time.sleep(0.1)

print("IA en action.")
try:
    while True:
        start_time = time.time()
        screen = grab_screen().resize(IMG_SIZE)
        input_img = np.expand_dims(np.array(screen) / 255.0, axis=0)

        prediction = model.predict(input_img, verbose=0)[0]
        apply_action_to_gamepad(prediction)

        elapsed = time.time() - start_time
        print(f"Temps écoulé pour la prédiction et l'action : {elapsed:.4f} secondes \n #######################################")
        time.sleep(max(0, FRAME_DELAY - elapsed))
except KeyboardInterrupt:
    print("Arrêt manuel.")
    reset_gamepad()
