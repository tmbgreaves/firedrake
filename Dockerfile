# DockerFile for a firedrake + jupyter container

# Use a jupyter notebook base image
FROM jupyter/scipy-notebook

# This DockerFile is looked after by
MAINTAINER Tim Greaves <tim.greaves@imperial.ac.uk>

# Update and install required packages for Firedrake
USER root
RUN apt-get update \
    && apt-get -y dist-upgrade \
    && apt-get -y install curl tzdata vim \
                 openssh-client build-essential autoconf automake \
                 cmake gfortran git libblas-dev liblapack-dev \
                 libopenmpi-dev libtool mercurial openmpi-bin \
                 python3-dev python3-pip python3-tk python3-venv \
                 zlib1g-dev libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# As the default session user, create a firedrake environment based on the 
# default jhub-scipy environment but without hdf5
USER jovyan
RUN pip install mpltools
RUN conda list --explicit | grep -v hdf5 | grep -v h5py | grep -v asn1crypto > spec-file.txt
RUN conda create -n firedrake --file spec-file.txt
RUN rm spec-file.txt

# Now install firedrake
WORKDIR /opt/conda/envs/firedrake/
RUN curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
RUN bash -c "source activate firedrake && python3 firedrake-install --no-package-manager --disable-ssh --install pyadjoint --install thetis --venv-name=firedrake-venv"

# Set up a bash environment for any interactive shell use
RUN mkdir -p /opt/conda/envs/firedrake/etc/conda/activate.d
RUN echo export VIRTUAL_ENV_DISABLE_PROMPT=1 > /opt/conda/envs/firedrake/etc/conda/activate.d/00_virtual_env_disable_prompt.sh
RUN cp /opt/conda/envs/firedrake/firedrake-venv/bin/activate /opt/conda/envs/firedrake/etc/conda/activate.d/01_firedrake-venv.sh
RUN mkdir -p /opt/conda/envs/firedrake/etc/conda/deactivate.d
RUN echo deactivate > /opt/conda/envs/firedrake/etc/conda/deactivate.d/01_firedrake-venv.sh
RUN echo "export PATH=$(echo $PATH | sed 's@/opt/conda/envs/firedrake/firedrake-venv/bin:@@')" >> /opt/conda/envs/firedrake/bin/deactivate

# Install an iPython kernel for firedrake
RUN bash -c "source activate firedrake && pip install jupyterhub ipykernel"
USER root
RUN mkdir -p /usr/local/share/jupyter
RUN chown jovyan /usr/local/share/jupyter
USER jovyan
RUN bash -c "source activate firedrake && python -m ipykernel install --name firedrake --display-name 'Python 3 (firedrake)'"

# Complete the environment, and leave the container in a state ready for jhub
RUN bash -c "source activate firedrake && pip install mpltools"
WORKDIR /home/jovyan
RUN cp -r /opt/conda/envs/firedrake/firedrake-venv/src/firedrake/docs/notebooks/* .
RUN rmdir work
ENV OMPI_MCA_btl=tcp,self
