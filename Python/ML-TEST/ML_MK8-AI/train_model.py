import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# GPU check
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU detected!")
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=20480)])
else:
    print("No GPU detected.")

# Chargement des données pour une session unique de ParcBaby
def load_data(csv_file, image_folder):
    data = pd.read_csv(csv_file, delimiter=';')
    images, labels = [], []
    for _, row in data.iterrows():
        image_path = os.path.join(image_folder, row.iloc[0])
        image = load_img(image_path, target_size=(64, 64))
        image = img_to_array(image) / 255.0
        images.append(image)
        labels.append([row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7]])
    return np.array(images), np.array(labels)

# Création du modèle
def create_cnn_model(input_shape=(64, 64, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Courbe de Perte')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Courbe de Précision')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Entraînement
def train_model():
    image_folder = '../../../Screens_AS/ParcBaby/'  # Dossier spécifique pour ParcBaby
    
    # Charger les fichiers CSV de la session ParcBaby
    csv_files = [f for f in os.listdir(image_folder) if f.endswith('.csv')]

    all_images, all_labels = [], []

    print(f"{len(csv_files)} fichiers CSV trouvés pour ParcBaby. Fusion en cours...")

    for csv_file in csv_files:
        csv_path = os.path.join(image_folder, csv_file)
        print(f"Chargement de : {csv_path}")
        images, labels = load_data(csv_path, image_folder)
        all_images.append(images)
        all_labels.append(labels)

    # Fusion des données
    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"{images.shape[0]} images et {labels.shape[0]} labels chargés au total pour ParcBaby.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Chargement ou création du modèle
    if os.path.exists('final_model.h5'):
        print("Modèle existant trouvé. Chargement pour entraînement continu...")
        model = load_model('final_model.h5')
    else:
        print("Création d’un nouveau modèle.")
        model = create_cnn_model()

    # Entraînement
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[checkpoint])

    # Évaluation
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    model.save('final_model.h5')

    # Affichage des courbes
    plot_history(history)


if __name__ == '__main__':
    train_model()
