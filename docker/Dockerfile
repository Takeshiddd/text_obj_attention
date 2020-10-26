FROM pytorch/pytorch:latest
RUN apt-get update
ARG USERNAME="kouki"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y --no-install-recommends \
	git \
        libsm6 \
        libxext6 \
        libgtk2.0-dev \
        language-pack-ja-base \
        language-pack-ja \
        libblas-dev \
        liblapack-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN apt-get update
RUN apt-get install -y curl wget
RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN rm get-pip.py
RUN apt-get install -y gcc
    
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-dev \
    && pip install opencv-python


RUN pip install torch==1.4.0

RUN pip install argparse && pip install torchvision==0.5.0 \
    && pip install matplotlib && pip install nltk \
    && pip install scikit-image \
    && pip install scipy && pip install Pillow-PIL \
    && pip install glob3 && pip install scikit-learn \
    && pip install tqdm && pip install gensim \ 
    && pip install faiss-gpu && pip install python-Levenshtein \ 
    && pip install visdom && pip install pycocotools


RUN apt-get install -y python3-tk && apt-get install -y libx11-dev
ENV DISPLAY host.docker.internal:0.0

WORKDIR /home/$USERNAME
RUN cd /home/$USERNAME
RUN python -m nltk.downloader stopwords

CMD ['nohup', 'python', '-m', 'visdom.server', '-p', '8097', '&']