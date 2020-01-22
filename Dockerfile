FROM continuumio/miniconda

RUN apt-get update -y
RUN apt-get install wget -y

RUN conda install pip -y

RUN pip install pyspark -y