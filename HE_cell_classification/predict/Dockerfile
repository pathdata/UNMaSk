FROM tensorflow/tensorflow:1.4.0-gpu-py3
MAINTAINER PN <priya.narayanan@icr.ac.uk>
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
run apt-get update && apt-get install --no-install-recommends -y \
	libsm6 \
	libxext6 \
	libxrender-dev

#################
# Python        #
#################
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#### Setup code and directories
# TensorBoard
EXPOSE 6006
# IPython/Jupyter
# EXPOSE 8888
