import os
import time
import numpy as np
import traceback
import threading
import uvicorn
from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import gc
import json
from typing import Optional

from routers.drawutility import *
from routers.image_utility import *

import torch

from models.experimental import attempt_load
from utils.general import non_max_suppression

DEFAULT_WEIGHT_FILE_PATH = os.environ.get("DEFAULT_WEIGHT", os.path.join('weights', 'yolov7-tiny.pt'))
print("default weight path {}".format(DEFAULT_WEIGHT_FILE_PATH), flush=True)
current_model_name = "yolov7-tiny"

lock = threading.Lock()


model = None
device = None
# conf_thres = 0.25
# iou_thres = 0.45
# print('conf_thres', conf_thres)
# print('iou_thres', iou_thres)


class StructureBase(BaseModel):
    conf_thres: Optional[float] = 0.25
    iou_thres: Optional[float] = 0.45
    model_name: Optional[str] = 'yolov7-tiny'

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


def clean_model():
    global model, device
    if model is not None:
        try:
            del model
            torch.cuda.memory_cached()
            torch.cuda.empty_cache()
            gc.collect()
            model = None
        except:
            print("fail to delete model")
    pass


def load_model(model_input):
    """
    model_input can be a local-path or a binary data of file uploaded by REST-ful API.
    the function should return all class name of models.
    """
    global model, device
    model = attempt_load(model_input, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    print('names', names)
    model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    model.to(device)
    return names


def initialize_model():
    global model, device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device', device)
    load_model(DEFAULT_WEIGHT_FILE_PATH)


initialize_model()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs2", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    local js, css 
    :return:
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get("/")
def HelloWorld():
    return {"Hello": "World"}


@app.post("/od_inference")
def yolov7_inference( data: StructureBase = Form(...), file: UploadFile = File(...)):
    # def yolov7_inference(file: bytes = File(...)):
    global device, model
    t0 = time.time()
    image_rgb = bytes_to_rgbimage(file.file.read())
    imgsize = image_rgb.shape[:2]
    image_rgb = image_rgb[:int(imgsize[0] / 32) * 32, :int(imgsize[1] / 32) * 32]
    image_rgb = np.transpose(image_rgb, (2, 0, 1))
    with lock:
        replace_model(data.model_name)
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        image_rgb = torch.from_numpy(image_rgb).to(device)
        image_rgb = image_rgb.float()  # uint8 to fp32
        image_rgb /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image_rgb.ndimension() == 3:
            image_rgb = image_rgb.unsqueeze(0)
        print('image_rgb.shape', image_rgb.shape)
        pred = model(image_rgb)[0]
        pred = non_max_suppression(pred, conf_thres=data.conf_thres, iou_thres=data.iou_thres)[0].cpu().numpy().tolist()

    print('pred', pred)
    t1 = time.time()
    fps = 1.0 / (t1 - t0)
    """
    output desc:
    {
    "detections": [[x1,y1,x2,y2,confidence,label_index],...],
    "class": a class label list,
    "fps": a float
    }
    output example:
    {
      "detections": [
        [263, 280.75, 400, 602, 0.91455078125, 0 ],
        [392.25, 315.5, 456.75, 476, 0.90087890625, 0]
      ],
      "class": ["person", "bicycle"],
      "fps": 3.6467737433932013
    }    
    """
    return {"detections": pred, "class": names, 'fps': fps}


def replace_model(model_name='yolov7-tiny'):
    global current_model_name
    try:
        if current_model_name != model_name:
            pt_filename = os.path.join('weights', model_name+'.pt')
            if os.path.isfile(pt_filename):
                print(f"loading {pt_filename}")
                current_model_name = model_name
                clean_model()
                names = load_model(pt_filename)
    except Exception as e:
        print(traceback.format_exc())
        restore_model()
    pass

@app.get("/restore_model")
def restore_model():
    """
    restore as default weight file
    :return:
    """
    t0 = time.time()
    clean_model()
    names = load_model(DEFAULT_WEIGHT_FILE_PATH)
    t1 = time.time()
    fps = 1.0 / (t1 - t0)
    return {"class": names, 'fps': fps}


@app.get("/get_class")
def get_class():
    t0 = time.time()
    global model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    print('names', names)
    t1 = time.time()
    fps = 1.0 / (t1 - t0)
    return {"class": names, 'fps': fps}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)
