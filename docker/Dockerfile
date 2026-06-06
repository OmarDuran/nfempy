ARG UBUNTU_VER=latest
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.12

FROM ubuntu:${UBUNTU_VER}
WORKDIR /app

# System packages 
ENV PYTHONDONTWRITEBYTECODE=true
ENV TZ=Europe/Oslo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update 
RUN apt-get install -yq curl vim wget git build-essential libopenblas-dev libtbb-dev
RUN apt-get install -yq libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 ffmpeg libsm6 libxext6 xvfb
RUN apt-get install -yq texlive dvipng texlive-fonts-recommended texlive-fonts-extra cm-super

ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${OS_TYPE}.sh"
RUN bash Miniforge3-Linux-${OS_TYPE}.sh -b -p /miniconda
RUN rm Miniforge3-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda config --add channels conda-forge
RUN conda update -n base -c conda-forge conda -y


ARG PY_VER
# Install packages from conda-forge and downgrade py (optional).
RUN conda install -c conda-forge -y python=${PY_VER}
RUN conda install -c conda-forge -y petsc4py gmsh

RUN git clone https://github.com/OmarDuran/nfempy.git
RUN git clone https://github.com/PTNobel/AutoDiff.git

RUN python -m pip install --upgrade pip
RUN python -m pip install pytest colorcet

WORKDIR /app/AutoDiff
RUN python -m pip install .

WORKDIR /app/nfempy
WORKDIR /app/nfempy
RUN grep -Ev "^(gmsh|mkl|mkl-service|pypardiso)" requirements.txt | python -m pip install --upgrade -r /dev/stdin
RUN python -m pip install --user -e .
RUN python -m pytest


