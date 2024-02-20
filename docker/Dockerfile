ARG UBUNTU_VER=latest
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.11

FROM ubuntu:${UBUNTU_VER}
WORKDIR /app

# System packages 
ENV PYTHONDONTWRITEBYTECODE=true

RUN apt-get update 
RUN apt-get install -yq curl wget git build-essential libopenblas-dev 
RUN apt-get install -yq libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 ffmpeg libgl1-mesa-glx libsm6 libxext6

ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
#RUN conda update -y conda
RUN conda config --add channels conda-forge
RUN conda update -n base -c conda-forge conda -y


ARG PY_VER
# Install packages from conda-forge and downgrade py (optional).
RUN conda install -c conda-forge -y python=${PY_VER}
RUN conda install -c conda-forge -y petsc4py 

RUN git clone https://github.com/OmarDuran/nfempy.git
WORKDIR /app/nfempy

RUN python -m pip install --upgrade pip \
    && python -m pip install pytest \
    && python -m pip install --upgrade -r requirements.txt \
    && python -m pip install .


RUN python -m pytest

# ENTRYPOINT ["tini", "-g", "--"]

