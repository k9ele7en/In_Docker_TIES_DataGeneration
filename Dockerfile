FROM tensorflow/tensorflow:latest
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install ffmpeg libsm6 libxext6 jq wget -y

# Firefox browser to run the tests
RUN apt-get install -y firefox
 
# Gecko Driver
ENV GECKODRIVER_VERSION 0.29.1
RUN wget --no-verbose -O /tmp/geckodriver.tar.gz https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux64.tar.gz \
  && rm -rf /opt/geckodriver \
  && tar -C /opt -zxf /tmp/geckodriver.tar.gz \
  && rm /tmp/geckodriver.tar.gz \
  && mv /opt/geckodriver /opt/geckodriver-$GECKODRIVER_VERSION \
  && chmod 755 /opt/geckodriver-$GECKODRIVER_VERSION \
  && ln -fs /opt/geckodriver-$GECKODRIVER_VERSION /usr/bin/geckodriver \
  && ln -fs /opt/geckodriver-$GECKODRIVER_VERSION /usr/bin/wires

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
# RUN apt-get install 'ffmpeg'\
#     'libsm6'\ 
#     'libxext6'  -y
CMD [ "/bin/bash" ]