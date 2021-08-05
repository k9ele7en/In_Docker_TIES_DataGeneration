FROM tensorflow/tensorflow:latest-jupyter
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get update
 
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
# RUN apt-get install 'ffmpeg'\
#     'libsm6'\ 
#     'libxext6'  -y
CMD [ "/bin/bash" ]