SERVER_CONTAINER="server"
sudo docker build . -t $SERVER_CONTAINER
PWD=$(pwd)

# Stop and delete the container if it exists
if [ "$(sudo docker ps -a -q -f name=$SERVER_CONTAINER)" ]; then
    if [ "$(sudo docker ps -aq -f status=running -f name=$SERVER_CONTAINER)" ]; then
        # cleanup
        sudo docker stop $SERVER_CONTAINER
    fi

    if [ "$(sudo docker ps -aq -f status=restarting -f name=$SERVER_CONTAINER)" ]; then
        # cleanup
        sudo docker stop $SERVER_CONTAINER
    fi

    # delete container
    sudo docker rm -f $SERVER_CONTAINER
fi

# Configure the container runtime
# sudo nvidia-ctk runtime configure --runtime=docker
# Run the server
sudo docker run --gpus all -d -p 12023:12023 --name $SERVER_CONTAINER nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
sudo docker ps
sudo docker logs $SERVER_CONTAINER