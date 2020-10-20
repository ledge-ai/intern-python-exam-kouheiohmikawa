FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY poetry.lock pyproject.toml ./

RUN pip install poetry \
	&& poetry config virtualenvs.create false \
	&& poetry install


RUN apt update && apt -y upgrade && apt install -y \
  build-essential \
  cmake \
  git \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev


# Install LightGBM
RUN git clone --recursive https://github.com/microsoft/LightGBM && cd LightGBM \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j4 

RUN cd LightGBM/python-package \
  && python setup.py install

RUN rm -r -f LightGBM/
