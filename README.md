# MK8 Deluxe AI Project
**MK8 AI** is an experimental project that aims to train an artificial intelligence to play Mario Kart 8 Deluxe autonomously on (*Yuzu*) emulator.

The system uses a **hybrid learning approach**:

- Supervised learning from recorded gameplay sessions (screens + controller inputs).

- Reinforcement learning in “VS Race” mode to improve performance dynamically.

The project includes:

- Real-time screen capture and input synchronization
- Virtual controller management via pyvjoy
- Neural network training (TensorFlow / Keras)
- Custom UI signal detection for race start/end
- Automated play and performance monitoring
