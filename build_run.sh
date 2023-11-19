sudo docker build . -t server
PWD=$(pwd)
sudo docker run -d -p 80:80 --name server --restart unless-stopped server
sudo docker ps
sudo docker logs server