FROM cschranz/gpu-jupyter:v1.3_cuda-10.1_ubuntu-18.04_python-only

USER root

RUN apt update \
 && apt install -y cmake libopenmpi-dev python3-dev zlib1g-dev swig python-opengl xvfb \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install stable_baselines3 stable_baselines3[extra] gym[box2d] pyvirtualdisplay PyOpenGL PyOpenGL-accelerate

USER jovyan
