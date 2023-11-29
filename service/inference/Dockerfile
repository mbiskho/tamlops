FROM kinlian/tamlops-prometheus:latest
LABEL service="inference"

#ARG
ARG REGISTRY_URI=""

#ENV
ENV REGISTRY_URI="${REGISTRY_URI}"

RUN apt-get update -y && apt-get install -y python3-pip
RUN apt-get install -y git

# requirement lib
COPY requirements.txt /
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt 
RUN pip3 install git+https://github.com/huggingface/diffusers
RUN pip3 install scikit-learn
RUN wandb login a00aa0dc4d721e8bd21428774471713f7a9c9141

# ## final configure
COPY . /app
WORKDIR /app

# Copy your Prometheus configuration file
# COPY prometheus.yml /etc/prometheus/prometheus.yml

RUN mkdir -p /etc/inference.d
COPY run.sh /etc/inference.d/run.sh
RUN chmod +x /etc/inference.d/run.sh

EXPOSE 8000
# EXPOSE 9100 

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl --fail http://127.0.0.1:8000/ || exit 1
CMD ["/etc/inference.d/run.sh"]
# CMD ["/etc/inference.d/run.sh", "--config.file=/etc/prometheus/prometheus.yml"]