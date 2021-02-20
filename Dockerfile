FROM nvcr.io/nvidia/tensorflow:19.03
RUN apt-get update
RUN apt-get install -y octave

FROM python:3.8

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

#COPY app/requirements_verbose.txt /app/requirements_verbose.txt


#copies the applicaiton from local path to container path
ADD . /app
WORKDIR /app

RUN pip3 install -r requirements.txt


ENV NUM_EPOCHS=10
ENV MODEL_TYPE='EfficientDet'
ENV DATASET_LINK='HIDDEN'
ENV TRAIN_TIME_SEC=100

CMD ["python3", "app.py"]