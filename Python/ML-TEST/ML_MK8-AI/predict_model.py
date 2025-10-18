import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

def predict_from_image(image_path):
    model = load_model('final_model.h5')
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]
    stick_x, stick_y = prediction[0], prediction[1]
    buttons = {
        'B': round(prediction[2]),
        'A': round(prediction[3]),
        'X': round(prediction[4]),
        'ZL': round(prediction[5]),
        'ZR': round(prediction[6])
    }

    print(f"Stick: X={stick_x:.2f}, Y={stick_y:.2f}")
    print("Boutons:", buttons)

if __name__ == '__main__':
    # Exemple d’image test (mets une image réelle ici)
    test_image = '../../../Screens_AS/test_predict.png'
    predict_from_image(test_image)
