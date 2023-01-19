FROM python:3.7-slim

EXPOSE 5000

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY download_weights.sh /app/
RUN bash download_weights.sh

ADD ./requirements.txt /app
RUN pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r requirements.txt

COPY ./*.py /app/
COPY ./routers /app/routers
COPY ./models /app/models
COPY ./utils /app/utils
COPY ./static /app/static

ENV PORT 5000
CMD uvicorn fastapi_interface:app --host=0.0.0.0 --port=${PORT}
