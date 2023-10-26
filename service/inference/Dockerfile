FROM bitnami/minideb:latest
LABEL service="inference"

## pre-requisite
RUN install_packages \
      ca-certificates \
      curl \
      git \
      gnupg \
      jq \
      make \
      openssh-client \
      python3 \
      python3-distutils \
      unzip \
      wget

RUN wget -q "https://bootstrap.pypa.io/get-pip.py" -O /tmp/get-pip.py \
      && python3 /tmp/get-pip.py --break-system-packages \
      && pip install netaddr --break-system-packages

# requirement lib
COPY requirements.txt /
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt --break-system-packages

# ## final configure
COPY . /app
WORKDIR /app

RUN mkdir -p /etc/inference.d
COPY run.sh /etc/inference.d/run.sh
RUN chmod +x /etc/inference.d/run.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl --fail http://127.0.0.1:8000/ || exit 1
CMD ["/etc/inference.d/run.sh"]