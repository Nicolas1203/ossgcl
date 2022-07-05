# Pull base pytorch image
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
# Install repository specific dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt