# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# install app dependencies
#RUN apt-get -y update && apt -get install software-properties-common \
#&& add-apt-repository ppa:deadsnakes/ppa && apt install python3.10.12
#RUN apt-get install -y python3-pip
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y libglew-dev libglew2.2 libglfw3 patchelf libglib2.0-0 xvfb

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml ./

#COPY pyproject.toml poetry.lock ./
#RUN touch README.md

#RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR
#RUN poetry install --without dev

RUN poetry install



ENTRYPOINT ["poetry", "run", "python", "-m"]