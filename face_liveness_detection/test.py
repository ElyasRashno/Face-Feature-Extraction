from mtcnn import MTCNN
import cv2
from detect_faceliveness import detect_face_liveness
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


folder = "/home/user/Desktop/face-biometric-server/data/face_male/"
images = load_images_from_folder(folder)
aa =len(images)
live=0;
spof=0;
facenotrec=0;


def detect_faces(img_npy):
    """
    Parameters: img_npy (a numpy array of an image)
    Output: the coordinate of two corner of bounding box (x1,y1,x2,y2) (top left point and bottom right points)

    This function uses MTCNN (multitask cascaded convolutional networks)
    You can find more info about this model in the link below
    https://github.com/ipazc/mtcnn
    """
    gray = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB)
    faces = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in
             [face["box"] for face in detector.detect_faces(gray)]]
    return faces

for i in range (len(images)):
    #img = cv2.imread("/home/amin/PycharmProjects/Face-Biometric-Server/data/single_face/man_2.jpeg")
    print("-"*50)
    print(os.listdir(folder)[i])
    img = img = images[i]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (detect_faces(img)):
        print("Face found")
        # break
        faces = detect_faces(img)
        # print(detect_face_cover(img, [box]))
        pred = detect_face_liveness(img, faces)
        print(pred)
        if (pred[0]=="live"):
            live += 1
        else:
            spof += 1
    else:
        print("Face Not Found")
        #a = 'Not Face'
        facenotrec += 1;
    #print(a)
print("live count: ", live)
print("spof count: ", spof)
print("Face Not Recognized: ", facenotrec)

