import cv2
from PIL import Image
from io import BytesIO
import base64
import numpy as np


def bytes_to_cv2image(imgdata):
    cv2img = cv2.cvtColor(np.array(Image.open(BytesIO(imgdata))), cv2.COLOR_RGB2BGR)
    return cv2img


def bytes_to_rgbimage(imgdata):
    cv2img = np.array(Image.open(BytesIO(imgdata)))
    return cv2img


def cv2image_to_base64(cv2img):
    retval, buffer_img = cv2.imencode('.jpg', cv2img)
    base64_str = base64.b64encode(buffer_img)
    str_a = base64_str.decode('utf-8')
    return str_a


def text2image(image, xy, label, font_scale=0.5, thickness=1, font_color=(0, 0, 0),
               font_face=cv2.FONT_HERSHEY_COMPLEX, background_color=(0, 255, 0)):
    label_size = cv2.getTextSize(label, font_face, font_scale, thickness)
    _x1 = xy[0]  # bottomleft x of text
    _y1 = xy[1]  # bottomleft y of text
    _x2 = xy[0] + label_size[0][0]  # topright x of text
    _y2 = xy[1] - label_size[0][1]  # topright y of text
    cv2.rectangle(image, (_x1, _y1), (_x2, _y2), background_color, cv2.FILLED)  # text background
    cv2.putText(image, label, (_x1, _y1), font_face, font_scale, font_color,
                thickness, cv2.LINE_AA)
