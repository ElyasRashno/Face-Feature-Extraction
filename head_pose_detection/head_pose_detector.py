import os
import cv2
import sys
# sys.path.append('..')


import numpy as np
from lib.FSANET_model import *
from keras import backend as K
from keras.layers import Average
from keras.models import Model

K.set_learning_phase(0)

# load model and weights
img_size = 64
stage_num = [3, 3, 3]
lambda_d = 1
detected = ''  # make this not local variable
ad = 0.6

num_capsule = 3
dim_capsule = 16
routings = 2
stage_num = [3, 3, 3]
lambda_d = 1
num_classes = 3
image_size = 64
num_primcaps = 7 * 3
m_dim = 5
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

model1 = FSA_net_Capsule(image_size, num_classes,
                         stage_num, lambda_d, S_set)()
model2 = FSA_net_Var_Capsule(
    image_size, num_classes, stage_num, lambda_d, S_set)()

num_primcaps = 8 * 8 * 3
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

model3 = FSA_net_noS_Capsule(
    image_size, num_classes, stage_num, lambda_d, S_set)()

print('Loading models ...')

weight_file1 = 'pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
model1.load_weights(weight_file1)
print('Finished loading model 1.')

weight_file2 = 'pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
model2.load_weights(weight_file2)
print('Finished loading model 2.')

weight_file3 = 'pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
model3.load_weights(weight_file3)
print('Finished loading model 3.')
print("Test Git")
inputs = Input(shape=(64, 64, 3))
x1 = model1(inputs)  # 1x1
x2 = model2(inputs)  # var
x3 = model3(inputs)  # w/o
avg_model = Average()([x1, x2, x3])
model = Model(inputs=inputs, outputs=avg_model)


def detect_head_rotation(detected_faces, input_img, faces, ad, img_size, img_w, img_h, model):
    heads_rotation = []

    for i, d in enumerate(detected_faces):
        # x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        x1, y1, x2, y2 = d

        w = x2 - x1
        h = y2 - y1

        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)

        faces[i, :, :, :] = cv2.resize(
            input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        faces[i, :, :, :] = cv2.normalize(
            faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        face = np.expand_dims(faces[i, :, :, :], axis=0)
        p_result = model.predict(face)

        yaw = p_result[0][0]
        pitch = p_result[0][1]
        roll = p_result[0][2]

        heads_rotation.append((yaw, pitch, roll))

    return heads_rotation


def detect_head_prose(img, detected_faces):
    global model
    img_h, img_w, _ = np.shape(img)

    # print(f"detected faces {len(detected_faces)}")
    # print(f"img {img}")

    faces = np.empty((len(detected_faces), img_size, img_size, 3))

    heads_rotation = detect_head_rotation(
        detected_faces, img, faces, ad, img_size, img_w, img_h, model)

    return heads_rotation
