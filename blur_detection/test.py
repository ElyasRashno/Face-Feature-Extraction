import cv2
from mtcnn import MTCNN
from blur_detector import detect_face_blur, add_padding_to_Face_border
import time
import os

detector = MTCNN()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#img = cv2.imread("/home/amin/PycharmProjects/Face-Biometric-Server/data/single_face/face_blur_1.jpeg")


folder = "/home/user/Desktop/face-biometric-server/data/face_male_female/"
images = load_images_from_folder(folder)
aa =len(images)
blur=0;
normal=0;
facenotrec=0;
for i in range (len(images)):
    print("-"*50)
    print(os.listdir(folder)[i])
    print()
    img = images[i]

# cv2.imshow("Hello",img)
# cv2.waitKey()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imshow("Hello",img)
# cv2.waitKey()

    if (detector.detect_faces(img)):
        print("Face found")
        #break
        face = detector.detect_faces(img)[0]
        box = face['box']
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        new_box = add_padding_to_Face_border(img, box, 0.2)

        a = (detect_face_blur(img, [new_box]))
        if(a=='Blur'):
            blur+=1
        else:
            normal+=1
        box = 0
        new_box=0
        face=0
    else:
        print("Face Not Found")
        a='Not Face'
        facenotrec+=1;
    print(a)
print("Blur count: ",blur)
print("Normal count: ", normal)
print("Face Not Recognized: ", facenotrec)
# print(img[box[1]:box[3], box[0]:box[2]].shape)
# print(img[new_box[1]:new_box[3], new_box[0]:new_box[2]].shape)
#
    #cv2.imshow("First", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
    #cv2.waitKey()
    #cv2.imshow("Second", img[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]])
    #cv2.waitKey()
