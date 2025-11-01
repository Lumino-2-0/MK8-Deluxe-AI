# MK8 AI
## THIS PROJECT IS NOT FINISHED!
MK8 AI is a hybrid machine learning project that teaches an AI to play *Mario Kart 8 Deluxe* on the Yuzu emulator.

---

## Overview

The project combines supervised learning (based on recorded gameplay sessions) and reinforcement learning to build a self-improving racing AI.

It captures screenshots, maps controller inputs, and uses a trained neural network to predict real-time actions.

---

## Features

- Virtual controller control via pyvjoy  
- Screen capture & synchronization at 15 FPS  
- Deep learning model (TensorFlow/Keras)  
- Data collection and automated training  
- Real-time gameplay via Jouer.py  
- Race start/end detection using custom color signals  

---

## Project Structure

```
MK_AI_Proj/
├── Python/
│   ├── ML-TEST/
│   │   └── ML_MK8-AI/
│   │       ├── Jouer.py
│   │       ├── ScreenHooker.py
│   │       ├── VJoy_Checker.py
│   │       ├── record.py
│   │       ├── utils.py
│   │       ├── final_model.h5
│   │       └── ...
│
├── Screens_AS/    # Supervised learning data
├── Screens_RT/    # Reinforcement learning data
└── Models/        # Trained models
```

---

## Requirements

- Python 3.10+  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Pyvjoy (for virtual gamepad)  
- PIL (Pillow)


---

## Usage

1. Record gameplay data using your controller.  
2. Train the model using the supervised dataset.  
3. Launch Jouer.py to let the AI play in real-time on Yuzu.  

---

## License

This project is licensed under the MIT License. See LICENSE for details.

---

## Notes

This is an experimental project made for fun and AI learning purposes — it is not affiliated with Nintendo or Yuzu.
