FROM kinlian/tamlops-prometheus:latest
LABEL service="inference"

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
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/huggingface/diffusers
RUN pip3 install scikit-learn
RUN pip3 install transformers
RUN pip3 install accelerate
RUN pip3 install sentencepiece
# RUN wandb login a00aa0dc4d721e8bd21428774471713f7a9c9141

# ## final configure
COPY . /app
WORKDIR /app

RUN mkdir -p /etc/inference.d
COPY run.sh /etc/inference.d/run.sh
RUN chmod +x /etc/inference.d/run.sh

RUN python3 loader.py

EXPOSE 8000 9090 9100
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl --fail http://127.0.0.1:8000/ || exit 1