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

## Reference 
- [yolov7](https://github.com/WongKinYiu/yolov7)
