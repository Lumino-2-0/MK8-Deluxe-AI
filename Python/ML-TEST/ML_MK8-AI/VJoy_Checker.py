import time
import pyvjoy
import math

# === Initialisation de la manette virtuelle ===
j = pyvjoy.VJoyDevice(1)
j.reset()
j.set_axis(pyvjoy.HID_USAGE_RX, 16384)
j.set_axis(pyvjoy.HID_USAGE_RY, 16384)
# === Test des boutons ===
print("Test des boutons A, B, X, ZR, ZL")
for button_id in range(1, 6):  # Boutons 1 à 5
    j.set_button(button_id, 1)
    print(f"Bouton {button_id} activé")
    time.sleep(0.5)
    j.set_button(button_id, 0)

# === Test du stick gauche (cercle) ===
print("Test du stick gauche (cercle)")
for i in range(36):
    angle = i * 10 * (math.pi / 180)
    x = int(16384 + math.cos(angle) * 16384)  # Rx
    y = int(16384 + math.sin(angle) * 16384)  # Ry
    j.set_axis(pyvjoy.HID_USAGE_RX, x)
    j.set_axis(pyvjoy.HID_USAGE_RY, y)
    time.sleep(0.01)

# === Remise à zéro ===
j.reset()
j.set_axis(pyvjoy.HID_USAGE_RX, 16384)
j.set_axis(pyvjoy.HID_USAGE_RY, 16384)
print("Test terminé.")
