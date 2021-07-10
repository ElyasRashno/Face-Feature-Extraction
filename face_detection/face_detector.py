from mtcnn import MTCNN
import cv2

detector = MTCNN()


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

