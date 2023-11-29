FROM kinlian/tamlops-prometheus:latest
LABEL service="interface"

#ARG
ARG REGISTRY_URI=""
ARG DB_HOST=""

#ENV
ENV REGISTRY_URI="${REGISTRY_URI}"
ENV DB_HOST="${DB_HOST}"

RUN apt-get update -y && apt-get install -y python3-pip
RUN apt-get install -y git

# requirement lib
COPY requirements.txt /
RUN pip3 install -r requirements.txt 
RUN pip3 install scikit-learn

# run prometheus + node-exporter
RUN systemctl enable node_exporter
RUN systemctl start node_exporter
RUN systemctl enable prometheus
RUN systemctl start prometheus


# ## final configure
COPY . /app
WORKDIR /app

RUN mkdir -p /etc/pipeline.d
COPY run.sh /etc/pipeline.d/run.sh
RUN chmod +x /etc/pipeline.d/run.sh

EXPOSE 8000 9090 9100
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl --fail http://127.0.0.1:8000/ || exit 1
CMD ["/etc/pipeline.d/run.sh"]