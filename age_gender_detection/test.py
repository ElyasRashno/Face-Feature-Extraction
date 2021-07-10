# import cv2
# from mtcnn import MTCNN
# from age_gender_detection import detect_age_gender, add_padding_to_Face_border
# import time
#
# detector = MTCNN()
#
# #img = cv2.imread("/home/amin/PycharmProjects/Face-Biometric-Server/data/multiple_face/multiple_face_2.jpg")
# img = cv2.imread("/home/user/Desktop/face-biometric-server/data/face_mask/Google_0581.jpeg")
#
#
# # cv2.imshow("Hello",img)
# # cv2.waitKey()
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# cv2.imshow("Hello",img)
# #cv2.waitKey()
#
#
# face = detector.detect_faces(img)[0]
# box = face['box']
# box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
# new_box = add_padding_to_Face_border(img, box, 0.2)
#
# print(detect_age_gender(img, [box]))
#
# print(img[box[1]:box[3], box[0]:box[2]].shape)
# print(img[new_box[1]:new_box[3], new_box[0]:new_box[2]].shape)
#
# cv2.imshow("First", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
# #cv2.waitKey()
# cv2.imshow("Second", img[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]])
# cv2.waitKey()
#################################################################################################################
import os
import cv2
from mtcnn import MTCNN
from age_gender_detection import detect_age_gender, add_padding_to_Face_border

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
images = load_images_from_folder(folder)
aa =len(images)
male=0;
female=0;
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
        a = (detect_age_gender(img, [box]))
        if(a[1][0]=='Male'):
            male+=1
        if (a[1][0] == 'Female'):
            female+=1
        box = 0
        new_box=0
        face=0
    else:
        print("Face Not Found")
        a='Not Face'
        facenotrec+=1;
    print(a)
print("Male count: ",male)
print("Female count: ", female)
print("Face Not Recognized: ", facenotrec)
# print(img[box[1]:box[3], box[0]:box[2]].shape)
# print(img[new_box[1]:new_box[3], new_box[0]:new_box[2]].shape)
#
    #cv2.imshow("First", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
    #cv2.waitKey()
    #cv2.imshow("Second", img[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]])
    #cv2.waitKey()
