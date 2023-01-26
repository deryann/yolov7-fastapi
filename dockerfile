FROM python:3.7-slim AS compile-image

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

ADD ./requirements.txt /app
RUN pip3 install --upgrade pip
RUN --mount=type=cache,target=/root/.cache \
    pip3 install --user -r requirements.txt

COPY download_weights.sh /app/
RUN bash download_weights.sh

FROM python:3.7-slim AS build-image
COPY --from=compile-image /root/.local /root/.local
COPY --from=compile-image /app/weights /app/weights
ENV PATH=/root/.local/bin:$PATH

EXPOSE 5000
WORKDIR /app



CMD uvicorn fastapi_interface:app --host=0.0.0.0 --port=${PORT}

COPY ./*.py /app/
COPY ./routers /app/routers
COPY ./models /app/models
COPY ./utils /app/utils
COPY ./static /app/static

ENV PORT 5000
