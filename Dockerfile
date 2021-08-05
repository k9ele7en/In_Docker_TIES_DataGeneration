FROM tensorflow/tensorflow:latest-jupyter
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install ffmpeg libsm6 libxext6 jq -y

# install geckodriver
RUN INSTALL_DIR="/usr/local/bin"

RUN json=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest)
RUN url=$(echo "$json" | jq -r '.assets[].browser_download_url | select(contains("linux64"))')
RUN curl -s -L "$url" | tar -xz
RUN chmod +x geckodriver
RUN sudo mv geckodriver "$INSTALL_DIR"
RUN echo "installed geckodriver binary in $INSTALL_DIR"

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
# RUN apt-get install 'ffmpeg'\
#     'libsm6'\ 
#     'libxext6'  -y
CMD [ "/bin/bash" ]