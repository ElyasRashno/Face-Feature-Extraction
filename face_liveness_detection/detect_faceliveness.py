import utils
from PIL import Image

model = utils.load_pkl_file("random_forrest_clf.pkl")

number_points = 8
radius = 2


def detect_face_liveness(img, faces):
    predictions = []

    for face in faces:
        cropped_face = img[face[1]:face[3], face[0]:face[2], :]

        print(cropped_face.shape)

        pil_image = Image.fromarray(cropped_face).convert('L')
        pred = utils.predict_liveness_real_time(pil_image, model, number_points, radius)
        predictions.append(pred[0])

    return predictions
