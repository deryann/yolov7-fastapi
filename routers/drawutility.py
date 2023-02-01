import time 
import json


from pydantic import BaseModel
from fastapi import APIRouter
from fastapi import FastAPI, File, Form, UploadFile
import cv2 


from routers.image_utility import * 

router = APIRouter()

color_list = [(29, 178, 255),
              (168, 153, 44),
              (49, 210, 207),
              (243, 126, 162),
              (89, 190, 22),
              (207, 190, 23),
              (99, 112, 171),
              (194, 119, 227),
              (180, 119, 31),
              (40, 39, 214)]

_cfg = {
    "line_width": 3,
    "label_font_size": 1.5,
    "text_font_size": 0.5,
}




def draw_bbox(image_bgr, bbox_min, bbox_max, line_color, line_width=2):
    cv2.rectangle(image_bgr, bbox_min, bbox_max, line_color, line_width)


class bboxBase(BaseModel):
    detections: list
    class_names: list

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@router.post("/draw_bbox/")
def web_if_draw_bbox(parameter: bboxBase = Form(...), file: UploadFile = File(...)):
    global _cfg
    t0 = time.time()

    # get image
    cv2_img = bytes_to_cv2image(file.file.read())

    # get info
    object_list = parameter.detections
    label_list = parameter.class_names
    # print("object_list", object_list)
    # print("label_list", label_list)

    if len(object_list) == 0:
        t1 = time.time()
        f_fps = 1.0 / (t1 - t0)
        output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
        return output_dict

    # draw
    for object in object_list:
        box_xmin = int(object[0])
        box_ymin = int(object[1])
        box_xmax = int(object[2])
        box_ymax = int(object[3])

        confidence = object[4]
        label = label_list[int(object[5])]
        color = color_list[int(object[5] % len(color_list))]

        draw_bbox(cv2_img, (box_xmin, box_ymin), (box_xmax, box_ymax), color, line_width=_cfg['line_width'])
        od_str = label + '  ' + str(confidence)[:4]
        text2image(cv2_img, (box_xmin, box_ymin), od_str, font_scale=_cfg['label_font_size'],
                   font_color=(255, 255, 255), font_face=cv2.FONT_HERSHEY_DUPLEX,
                   background_color=color)

    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
    return output_dict


class polyBase(BaseModel):
    points: list = [[10, 90], [20, 90], [20, 100], [10, 100]]
    color: list = [0, 255, 0]
    thickness: int = 2
    is_closed: bool = True

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@router.post("/draw_poly/")
def draw_poly(parameter: polyBase = Form(...), file: UploadFile = File(...)):
    t0 = time.time()

    # get image
    cv2_img = bytes_to_cv2image(file.file.read())

    # get info
    object_list = parameter.points

    if len(object_list) == 0:
        t1 = time.time()
        f_fps = 1.0 / (t1 - t0)
        output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
        return output_dict

    pts = np.array(object_list)
    pts = pts.reshape((-1, 1, 2))

    cv2_img = cv2.polylines(cv2_img, [pts], parameter.is_closed, parameter.color, parameter.thickness)

    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
    return output_dict