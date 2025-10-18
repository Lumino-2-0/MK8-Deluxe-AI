# utils.py

import numpy as np
import math
from inputs import get_gamepad
import threading
import os

def ensure_dir_exists(directory):
    """
    Crée le dossier s'il n'existe pas déjà.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def normalize_stick_value(value, deadzone=8000, max_range=32768):
    """
    Normalise une valeur brute de stick analogique (-32768 à 32767) entre 0 et 1.
    """
    if abs(value) < deadzone:
        return 0.5  # Zone morte = centre
    norm = (value + max_range) / (2 * max_range)  # convertit -32768:32767 => 0:1
    return min(max(norm, 0), 1)

class XboxController(object):
    """Classe pour récupérer les valeurs de la manette Xbox"""

    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):
        # Initialisation des variables pour les boutons et sticks
        self.LeftJoystickY = 0.5
        self.LeftJoystickX = 0.5
        self.A = 0
        self.B = 0
        self.X = 0
        self.ZL = 0
        self.ZR = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(self):
        """Retourne les valeurs normalisées des inputs importants (stick gauche + A, B, X, ZL, ZR)"""
        return [
            round(self.LeftJoystickX, 3),
            round(self.LeftJoystickY, 3),
            self.A,
            self.B,
            self.X,
            self.ZL,
            self.ZR
        ]

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_X':
                    self.LeftJoystickX = normalize_stick_value(event.state)
                elif event.code == 'ABS_Y':
                    self.LeftJoystickY = normalize_stick_value(event.state)
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state
                elif event.code == 'ABS_Z':
                    self.ZL = 1 if event.state > 10 else 0
                elif event.code == 'ABS_RZ':
                    self.ZR = 1 if event.state > 10 else 0

def get_gamepad_input_normalized():
    controller = XboxController()
    return controller.read()
