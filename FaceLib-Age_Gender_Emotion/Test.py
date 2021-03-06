import matplotlib.pyplot as plt
from facelib import FaceRecognizer, FaceDetector
from facelib import update_facebank, load_facebank, special_draw, get_config
import cv2
import os
import matplotlib.pyplot as plt
#from facelib import FaceDetector
#from facelib import AgeGenderEstimator, FaceDetector
from facelib import FaceDetector, EmotionDetector, AgeGenderEstimator

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
#folder = "/home/user/Desktop/face-biometric-server/data/face_male"
folder = "/home/user/Desktop/face-biometric-server/data/rashionalrose/test"
#img = cv2.imread(os.path.join(folder,"Google_0413.jpeg"))
images = load_images_from_folder(folder)
age_gender_detector = AgeGenderEstimator()
emotion_detector = EmotionDetector()
#face_detector = FaceDetector(face_size=(224, 224))

aa =len(images)
male=0;
female=0;
# detector = FaceDetector()
face_detector = FaceDetector()
for i in range (len(images)):
    print("-"*50)
    print(os.listdir(folder)[i], "Num:", i)
    #print()
    img = images[i]
    #faces, boxes, scores, landmarks = detector.detect_align(img)


    faces, boxes, scores, landmarks = face_detector.detect_align(img)
    genders, ages = age_gender_detector.detect(faces)
    #print(genders, ages)

    #faces, boxes, scores, landmarks = face_detector.detect_align(img)
    list_of_emotions, probab = emotion_detector.detect_emotion(faces)
    print(genders, ages, list_of_emotions)
    if (genders[0]=="Male"):
        male+=1;
    if (genders[0]=="Female"):
        female+=1;

print("Male Count: ", male)
print("Female Count: ", female)
# img = plt.imread('facelib/imgs/face_rec.jpg')
# detector = FaceDetector()
# faces, boxes, scores, landmarks = detector.detect_align(img)
# #aa=faces.numpy()
# #plt.imshow(faces.cpu()[0]);
# #cv2.imwrite("FACE.jpg",faces[0])
# #cv2.imwrite("FACE.jpg",aa[0])
#
#
#
#
#
# img = plt.imread('facelib/imgs/face_rec.jpg')
# face_detector = FaceDetector()
# age_gender_detector = AgeGenderEstimator()
#
# faces, boxes, scores, landmarks = face_detector.detect_align(img)
# genders, ages = age_gender_detector.detect(faces)
# print(genders, ages)
#
#
# import matplotlib.pyplot as plt
#
#
# img = plt.imread('facelib/imgs/face_rec.jpg')
# face_detector = FaceDetector(face_size=(224, 224))
# emotion_detector = EmotionDetector()
#
# faces, boxes, scores, landmarks = face_detector.detect_align(img)
# list_of_emotions, probab = emotion_detector.detect_emotion(faces)
# print(list_of_emotions)
#
# conf = get_config()
# detector = FaceDetector()
# face_rec = FaceRecognizer(conf)
# face_rec.model.eval()
#
# update = True
# if update:
#     targets, names = update_facebank(conf, face_rec.model, detector)
# else:
#     targets, names = load_facebank(conf)
#
# img = plt.imread('facelib/imgs/face_rec.jpg')
#
# faces, boxes, scores, landmarks = detector.detect_align(img)
# results, score = face_rec.infer(conf, faces, targets)
# names[results.cpu()]
