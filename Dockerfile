# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# install app dependencies
#RUN apt-get -y update && apt -get install software-properties-common \
#&& add-apt-repository ppa:deadsnakes/ppa && apt install python3.10.12
#RUN apt-get install -y python3-pip
# Avoid warnings by switching to noninteractive for the build process
ENV DEBIAN_FRONTEND=noninteractive

ENV USER=root

RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y libopenmpi-dev libglew-dev libglew2.2 libglfw3 patchelf libglib2.0-0 xvfb

# Install XFCE, VNC server, dbus-x11, and xfonts-base
RUN apt-get install -y xfce4 xfce4-goodies tightvncserver dbus-x11 xfonts-base wget
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y ./google-chrome-stable_current_amd64.deb

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Setup VNC server
RUN mkdir /root/.vnc \
    && echo "password" | vncpasswd -f > /root/.vnc/passwd \
    && chmod 600 /root/.vnc/passwd

# Create an .Xauthority file
RUN touch /root/.Xauthority

# Set display resolution (change as needed)
ENV RESOLUTION=1920x1080

# Expose VNC port
EXPOSE 5901

WORKDIR /app

COPY pyproject.toml ./

RUN poetry install

# Copy a script to start the VNC server
COPY start-vnc.sh start-vnc.sh
RUN chmod +x start-vnc.sh

#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["poetry", "run", "python", "-m"]