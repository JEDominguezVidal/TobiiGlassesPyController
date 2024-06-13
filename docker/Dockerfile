FROM python:3.8
LABEL maintainer="Davide De Tommaso <davide.detommaso@iit.it>"

ARG TOBIIGLASSESCTRL_VER

RUN adduser --disabled-password docky

USER docky

WORKDIR /home/docky


ENV PATH=${PATH}:/home/docky/.local/bin

USER docky
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    && apt-get update && apt-get install -y \
    python3-gi \
    python3-gst-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    && rm -rf /var/lib/apt/lists/*

RUN cd $HOME &&\
    git clone --recursive https://github.com/ddetommaso/TobiiGlassesPyController &&\
    cd TobiiGlassesPyController &&\
    git checkout ${TOBIIGLASSESCTRL_VER} &&\
    pip3 install --user .
