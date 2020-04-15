FROM conda/miniconda3
RUN conda update -n base -c defaults conda

RUN apt-get update -y
RUN apt-get install openjdk-8-jdk -y
RUN apt-get install wget -y
RUN apt-get install git -y
RUN apt-get install libatlas3-base libopenblas-base -y
RUN apt-get install curl -y
RUN apt-get install gnupg2 -y
RUN apt-get install apt-transport-https -y

RUN echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
RUN curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add
RUN apt-get update
RUN apt-get install sbt

#RUN wget https://downloads.apache.org/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz
#RUN tar -xzf apache-maven-3.6.3-bin.tar.gz
#RUN mv apache-maven-3.6.3-bin /opt/maven-3.6.3
#RUN ln -s /opt/maven-3.6.3 /opt/maven
#ENV PATH="/opt/maven/bin:$PATH"

RUN conda install python=3.7
RUN conda install pip -y
RUN conda install -c bioconda scala

RUN git clone https://github.com/apache/spark.git /opt/spark-3.0.0 \
&& cd /opt/spark-3.0.0 \
&& git checkout tags/v3.0.0-preview2
RUN cd /opt/spark-3.0.0 && build/mvn -DskipTests -Pnetlib-lgpl clean package
RUN ln -s /opt/spark-3.0.0 /opt/spark
ENV SPARK_HOME="/opt/spark"
ENV PATH="/opt/spark/bin:$PATH"

#RUN pip install pyspark