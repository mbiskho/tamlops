# FROM ubuntu:latest
# LABEL service="pipeline"

# #ARG
# ARG REGISTRY_URI=""
# ARG DB_HOST=""

# #ENV
# ENV REGISTRY_URI="${REGISTRY_URI}"
# ENV DB_HOST="${DB_HOST}"

# RUN apt-get update -y && apt-get install -y python3-pip
# RUN apt-get install -y git

# # requirement lib
# COPY requirements.txt /
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip3 install -r requirements.txt 
# RUN pip3 install git+https://github.com/huggingface/diffusers
# RUN pip3 install scikit-learn
# RUN wandb login a00aa0dc4d721e8bd21428774471713f7a9c9141

# # ## final configure
# COPY . /app
# WORKDIR /app

# RUN mkdir -p /etc/pipeline.d
# COPY run.sh /etc/pipeline.d/run.sh
# RUN chmod +x /etc/pipeline.d/run.sh

# EXPOSE 8000
# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl --fail http://127.0.0.1:8000/ || exit 1
# CMD ["/etc/pipeline.d/run.sh"]

FROM nvcr.io/nvidia/pytorch:23.10-py3
LABEL service="pipeline"

#ARG
ARG REGISTRY_URI=""
ARG DB_HOST=""

#ENV
ENV REGISTRY_URI="${REGISTRY_URI}"
ENV DB_HOST="${DB_HOST}"

RUN apt-get update -y && apt-get install -y python3-pip
RUN apt-get install -y git

# requirement lib
COPY requirements.txt ./
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt 
RUN pip3 install git+https://github.com/huggingface/diffusers
RUN pip3 install scikit-learn
RUN wandb login a00aa0dc4d721e8bd21428774471713f7a9c9141

# ## final configure
COPY . /app
WORKDIR /app

RUN mkdir -p /etc/pipeline.d
COPY run.sh /etc/pipeline.d/run.sh
RUN chmod +x /etc/pipeline.d/run.sh
RUN chmod +x image.sh
RUN chmod +x text.sh
RUN chmod +x gpu_image.sh
RUN chmod +x gpu_text.sh

# install prometheus + node-exporter
# RUN mkdir -p /scripts
# COPY install-prom-node.sh /scripts/install-prom-node.sh
# RUN chmod +x /scripts/install-prom-node.sh
# RUN /scripts/install-prom-node.sh

EXPOSE 8000 9090
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl --fail http://127.0.0.1:8000/ || exit 1
CMD ["/etc/pipeline.d/run.sh"]