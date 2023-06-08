# Conda setup
FROM python:slim
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n python-mockup-streamer && \
    conda activate python-mockup-streamer && \
    conda install python=3.10 pip

# Specific to mockup_streamer
WORKDIR /usr/src/app

COPY mockup_streamer ./mockup_streamer
RUN mkdir config
COPY config/streaming_docker.toml ./config/streaming.toml
COPY requirements.txt .

RUN pip install -U pip
RUN pip install -r requirements.txt
RUN conda install -c conda-forge liblsl

COPY api/server.py ./
RUN chmod +x server.py


RUN ls -ahl
CMD ["python", "-m", "server"]
