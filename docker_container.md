### Docker Container and Installation Instructions

sudo docker build -t docker_gui_opentamp .


Running with GUI: 

sudo docker run -it --entrypoint /bin/sh -v {insert_local_opentamp_path_here}:/app/../opentamp/ -p 5901:5901 docker_gui_opentamp

Fully written out example: 

sudo docker run -it --entrypoint /bin/sh -v /home/rarama/Documents/research/OpenTAMP_python_alter/openTAMP:/app/../opentamp/ -p 5901:5901 docker_gui_opentamp


1) Run ./start-vnc.sh  Whatever remote desktop method, localhost:5901, password is ``password" or see Dockerfile if you like to change this. 
2) poetry shell (activate the environment)
3) cd into /app/../opentamp/ folder on the remote desktop, then ``pip install -e . " (to complete install opentamp module)
4) To open google chrome browser on VNC, have to do command: google-chrome --no-sandbox

