SERVER_CONTAINER="server"
sudo docker build . -t $SERVER_CONTAINER
PWD=$(pwd)

# Stop and delete the container if it exists
if [ "$(sudo docker ps -a -q -f name=$SERVER_CONTAINER)" ]; then
    if [ "$(sudo docker ps -aq -f status=running -f name=$SERVER_CONTAINER)" ]; then
        # cleanup
        sudo docker stop $SERVER_CONTAINER
    fi

    # delete container
    sudo docker rm $SERVER_CONTAINER
fi

# Run the server
sudo docker run -d -p 12023:12023 --name $SERVER_CONTAINER --restart unless-stopped $SERVER_CONTAINER
sudo docker ps
sudo docker logs $SERVER_CONTAINER