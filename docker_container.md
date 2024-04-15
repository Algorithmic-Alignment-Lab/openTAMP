### Docker Container and Installation Instructions

sudo docker build -t opentamptest:poetryversion . 


sudo docker run -it --entrypoint /bin/sh -v /home/rarama/Documents/research/OpenTAMP_python_alter/openTAMP:/app/../opentamp/ opentamptest:poetryversion

pip install -e .

pip install pyvirtualdisplay
