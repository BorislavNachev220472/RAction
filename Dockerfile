FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y ffmpeg libsm6 libxext6

RUN apt-get update && apt-get install python3-pip -y

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python3.9 -m pip install -r requirements.txt

COPY ./api ./app

EXPOSE 808
EXPOSE 22

CMD gunicorn 'app:app' --bind=0.0.0.0:808
