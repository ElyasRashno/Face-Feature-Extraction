import sys
sys.path.append("./")

import cv2
import numpy as np
import mxnet as mx
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector
from preproccessing.dataset_proc import gen_face, gen_boundbox

# from mtcnn import MTCNN
# detector = MTCNN()

MTCNN_DETECT = MtcnnDetector(model_folder=None, ctx=mx.cpu(0), num_worker=1, minsize=50, accurate_landmark=True)


#model_weight_path = "model/c3ae_model_v2_fp16_white_se_132_4.208622-0.973"
model_weight_path =  "model/c3ae_model_v2_fp16_white_se_132_4.208622-0.973"

def load_C3AE2():
    from C3AE_expand import build_net3, model_refresh_without_nan 
    models = build_net3()
    
    models.load_weights(model_weight_path)
    model_refresh_without_nan(models) ## hot fix which occur non-scientice gpu or cpu
    return models


model = load_C3AE2()

def predict(img):
    # try:
    #     bounds, lmarks = gen_face(MTCNN_DETECT, img, only_one=False)
    #     ret = MTCNN_DETECT.extract_image_chips(img, lmarks, padding=0.4)
    # except Exception as ee:
    #     ret = None
    #     print(img.shape, ee)
    # if not ret:
    #     print("no face")
    #     return None

    if(gen_face(MTCNN_DETECT, img, only_one=False)==False):
        print("NNNNNNNNNNNNNNN")
        return 0, 0
    else:
        bounds, lmarks = gen_face(MTCNN_DETECT, img, only_one=False)
        ret = MTCNN_DETECT.extract_image_chips(img, lmarks, padding=0.4)



    padding = 200
    new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    bounds, lmarks = bounds, lmarks
    
    colors = [(0, 0, 255), (0, 0, 0), (255, 0, 0)]
    for pidx, (box, landmarks) in enumerate(zip(bounds, lmarks)):
        trible_box = gen_boundbox(box, landmarks)
        tri_imgs = []
        for bbox in trible_box:
            bbox = bbox + padding
            h_min, w_min = bbox[0]
            h_max, w_max = bbox[1]
            #cv2.imwrite("test.jpg", new_bd_img[w_min:w_max, h_min:h_max, :])
            tri_imgs.append([cv2.resize(new_bd_img[w_min:w_max, h_min:h_max, :], (64, 64))])

        # for idx, pbox in enumerate(trible_box):
        #     pbox = pbox + padding
        #     h_min, w_min = pbox[0]
        #     h_max, w_max = pbox[1]
        #     new_bd_img = cv2.rectangle(new_bd_img, (h_min, w_min), (h_max, w_max), colors[idx], 2)


        print(f'tri_imgs {np.shape(tri_imgs)}')
        
        result = model.predict(tri_imgs)
        age, gender = None, None
        if result and len(result) == 3:
            age, _, gender = result
            age_label, gender_label = age[-1][-1], "Female" if gender[-1][0] > gender[-1][1] else "Male"
        elif result and len(result) == 2:
            age, _  = result
            age_label, gender_label = age[-1][-1], "unknown"
        else:
           raise Exception("fatal result: %s"%result)
        # cv2.putText(new_bd_img, '%s %s'%(int(age_label), gender_label), (padding + int(bounds[pidx][0]), padding + int(bounds[pidx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 2, 175), 2)
    # if save_image:
    #     print(result)
    #     cv2.imwrite("igg.jpg", new_bd_img)
    return (age_label, gender_label)

# img = cv2.imread("multiple_face_2.jpg")
# img = cv2.imread("../assets/timg.jpg")
# img = cv2.imread("man_face.jpeg")
# img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# print(predict(img))

def add_padding_to_Face_border(img, face, add_border_ratio=0.1):
    img_h, img_w, _ = np.shape(img)
    #img_w, img_h, _ = np.shape(img)
    #img_w = min(img_h,img_w)
    #img_h = min(img_h, img_w)
    # print(f"face {face}")

    x1, y1, x2, y2 = face

    w = x2 - x1
    h = y2 - y1

    new_x1 = int(max(0, x1 - add_border_ratio * w))
    new_y1 = int(max(0, y1 - add_border_ratio * h))
    new_x2 = int(min(img_w, x2 + add_border_ratio * w))
    new_y2 = int(min(img_h, y2 + add_border_ratio * h))

    return [new_x1, new_y1, new_x2, new_y2]


def detect_age_gender(img,faces):
    """
    detection of gender and age estimation for the faces in the image
    :param img: the image
    :param faces: the face boxes in the image. for each face we have box like this[x1,y1,x2,y2]
    :return: two list, the first one is the list of genders and list of ages for each crossponding face
    """
    
    age_lst = []
    gender_lst = []
    
    for i, face in enumerate(faces):
        
        # print(f"face_num {i+1}")
        face = add_padding_to_Face_border(img,face)
        cropped_img = img[face[1]:face[3],face[0]:face[2],:]
        
        # print(f"face {face}")
        # print(f"cropped_img_shape : {cropped_img.shape}")
        
        age, gender = predict(cropped_img)
        
        age_lst.append(age)
        gender_lst.append(gender)
        
    return age_lst, gender_lst
