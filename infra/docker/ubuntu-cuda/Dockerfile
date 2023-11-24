# Use the specific CUDA base image
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

# Update packages and install Python, pip, and git
RUN apt-get update && \
    apt-get install -y python3 python3-pip git

RUN pip3 install torch torchvision torchaudio

# Set an entrypoint to keep the container running
CMD ["tail", "-f", "/dev/null"]
