#
# #detector = MTCNN()
#
# #img = cv2.imread("/home/amin/PycharmProjects/Face-Biometric-Server/data/single_face_mask/face_mask_7.jpeg")
#
# # cv2.imshow("Hello",img)
# # cv2.waitKey()
#
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # cv2.imshow("Hello",img)
# # cv2.waitKey()
#
#
# face = detector.detect_faces(img)[0]
# box = face['box']
# box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
# new_box = add_padding_to_Face_border(img, box, 0.2)
#
# print(detect_face_cover(img, [box]))
#
# # print(img[box[1]:box[3], box[0]:box[2]].shape)
# # print(img[new_box[1]:new_box[3], new_box[0]:new_box[2]].shape)
# #
# cv2.imshow("First", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
# cv2.waitKey()
# cv2.imshow("Second", img[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]])
# cv2.waitKey()

import cv2
from mtcnn import MTCNN
from face_cover_detector import detect_face_cover, add_padding_to_Face_border
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


folder = "/home/user/Desktop/face-biometric-server/data/face_age/wiki_crop/02"
#img = cv2.imread(os.path.join(folder,"Google_0413.jpeg"))
images = load_images_from_folder(folder)
aa =len(images)
cover=0;
DarkGlasses=0;
NormalGlasses=0;
Covered_By_Hand=0;
Normal=0;
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

        #print(detect_face_cover(img, [box]))
        a = (detect_face_cover(img, [new_box]))
        if(a=='Mask'):
            cover+=1
        if(a=='DarkGlasses'):
            DarkGlasses+=1
        if(a=='NormalGlasses'):
            NormalGlasses+=1
        if(a=='Covered_By_Hand'):
            Covered_By_Hand+=1
        if(a=='Normal'):
            Normal+=1
        box = 0
        new_box=0
        face=0
    else:
        print("Face Not Found")
        a='Not Face'
        facenotrec+=1;
    print(a)
print("Cover count: ",cover)
print("DarkGlasses count: ",DarkGlasses)
print("NormalGlasses count: ",NormalGlasses)
print("Covered_By_Hand count: ",Covered_By_Hand)
print("Normal count: ", Normal)
print("Face Not Recognized: ", facenotrec)
# print(img[box[1]:box[3], box[0]:box[2]].shape)
# print(img[new_box[1]:new_box[3], new_box[0]:new_box[2]].shape)
#
    #cv2.imshow("First", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
    #cv2.waitKey()
    #cv2.imshow("Second", img[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]])
    #cv2.waitKey()
