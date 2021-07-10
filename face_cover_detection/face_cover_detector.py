from tensorflow import keras
import cv2
import numpy as np
import tensorflow as tf

model = keras.models.load_model("EfficientNet.h5")
# model.summary()

classes = {0: "DarkGlasses", 1: "NormalGlasses", 2: "Mask", 3: "Covered_By_Hand", 4: "Normal"}


def detect_face_cover(img, faces):
    covered_face = []

    for face in faces:
        print(face)
        face_img = img[face[1]:face[3], face[0]:face[2], :]
        face_img = cv2.resize(face_img, (224, 224))
        face_img.resize((1, 224, 224, 3))
        # tensor = tf.convert_to_tensor(face_img, dtype='int8')
        # pred_label = model.predict(tensor)
        pred_label = model(face_img)

        print(pred_label)

        pred_label = tf.math.argmax(pred_label, axis=-1)
        pred_label = classes[pred_label.numpy()[0]]
        prediction = pred_label
        covered_face.append(pred_label)
        return prediction

    return covered_face


def add_padding_to_Face_border(img, face, add_border_ratio):
    img_h, img_w, _ = np.shape(img)

    print(f"face {face}")

    x1, y1, x2, y2 = face

    w = x2 - x1
    h = y2 - y1

    new_x1 = int(max(0, x1 - add_border_ratio * w))
    new_y1 = int(max(0, y1 - add_border_ratio * h))
    new_x2 = int(min(img_w, x2 + add_border_ratio * w))
    new_y2 = int(min(img_h, y2 + add_border_ratio * h))

    return [new_x1, new_y1, new_x2, new_y2]
