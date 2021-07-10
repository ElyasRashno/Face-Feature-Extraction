from mtcnn.mtcnn import MTCNN
from head_pose_detector import detect_head_prose
import cv2
import mtcnn

detector = MTCNN()
img = cv2.imread("Google_0784.jpeg")

def detect_faces(img_npy):
    gray = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB)
    faces = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in
             [face["box"] for face in detector.detect_faces(gray)]]
    return faces

faces = detect_faces(img)
aa = detect_head_prose(img,faces)
print(detect_head_prose(img,faces))

