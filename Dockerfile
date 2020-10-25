FROM mariuswi/trajopt_ws:1.0
RUN pip install pybullet
WORKDIR /root/catkin_ws/src
