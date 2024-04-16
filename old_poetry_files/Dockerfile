FROM python:3.9-buster
RUN apt-get update && apt-get install -y \
libglfw3 \
libglew2.1 \
patchelf \
libopenmpi-dev
RUN pip install poetry
COPY . .
RUN pip install wheel==0.38.4 setuptools==65.5.1
RUN poetry install