FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 python3 >/dev/null 2>&1
RUN apt-get install -y python3-pip git wget curl ffmpeg libsm6 libxext6 gcc unzip g++ apt-utils >/dev/null 2>&1
RUN apt-get install lsb-release curl gpg -y >/dev/null 2>&1

RUN mkdir /ocrDemo

WORKDIR /ocrDemo


COPY requirements.txt .
RUN ls -lh

# ARG GITHUB_TOKEN

RUN pip install -r requirements.txt
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# RUN python -m pip install "git+https://x-access-token:${GITHUB_TOKEN}@github.com/Frinks-ai/detectron2"


