FROM ros:humble-ros-base

# Install MTCNN dependencies (adjust as needed)
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get update && apt-get install -y --fix-missing libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Copy your ROS2 node code
COPY . /workspace/src/my_mtcnn_node

# Build your ROS2 node
WORKDIR /workspace/src/my_mtcnn_node
RUN colcon build --symlink-install

# Copy requirements file first (this layer will be cached)
COPY requirements.txt .

# Install Python dependencies (this layer will be cached)
RUN pip install --no-cache-dir -r requirements.txt

# Run your node
#CMD ["ros2", "run", "my_mtcnn_node", "mtcnn_detector"]
