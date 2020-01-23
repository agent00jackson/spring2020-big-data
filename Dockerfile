FROM conda/miniconda3
RUN conda update -n base -c defaults conda

RUN apt-get update -y
RUN apt-get install openjdk-8-jdk -y
RUN apt-get install wget -y

RUN conda install python=3.7
RUN conda install pip -y
RUN conda install -c bioconda scala

RUN wget http://www.gtlib.gatech.edu/pub/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop3.2.tgz
RUN tar -xzf spark-3.0.0-preview2-bin-hadoop3.2.tgz
RUN mv spark-3.0.0-preview2-bin-hadoop3.2 /opt/spark-3.0.0
RUN ln -s /opt/spark-3.0.0 /opt/spark
ENV SPARK_HOME="/opt/spark"
ENV PATH="/opt/spark/bin:$PATH"

#RUN pip install pyspark