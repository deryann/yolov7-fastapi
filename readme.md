# Yolov7 Fast API
- Yolov7 inference code implemented by FastAPI

## Docker command reference 

### How to build it and run it.

```bash 
# build it
docker build -t yolov7fastapi:latest .

# run it (CPU)
docker run -it -p 127.0.0.1:5000:5000 yolov7fastapi:latest

# run it (GPU)
docker run -it -p 127.0.0.1:5000:5000 --gpus all yolov7fastapi:latest
```
## test restful api code 
```python
import json, requests, os 
import pprint

filename = "cup.jpg"
url = 'http://[IP]:[PORT]/od_inference'

payload = {
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "model_name": "yolov7-tiny"
    }

#payload = {"param_1": "value_1", "param_2": "value_2"}
files = {
    'data': (None, json.dumps(payload), 'application/json'),
    'file': (os.path.basename(filename), open(filename, 'rb'), 'application/octet-stream')
}
try:

    response = requests.post(url,files=files )
    if response.ok:
        t = response.json()
        pprint.pprint(t)
    else:
        print(f"[Error] You may not launch yolov7 api interface in {url}")
        t = dict()
except Exception as e:
    print(traceback.format_exc())
    t = dict()
    
```

## Reference 
- [yolov7](https://github.com/WongKinYiu/yolov7)
