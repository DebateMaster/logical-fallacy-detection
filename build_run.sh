sudo docker build . -t network
PWD=$(pwd)
sudo docker run -it -v $PWD:/workspace/network network