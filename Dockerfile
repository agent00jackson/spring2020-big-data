FROM conda/miniconda3

RUN apt-get install openjdk-8-jdk -y

RUN conda update -n base -c defaults conda
RUN apt-get update -y
RUN apt-get install wget -y

RUN conda install python=3.7
RUN conda install pip -y

RUN pip install pyspark