cd .. 
export LATEST_MODEL=$(ls -t models/ | head -1)

echo $LATEST_MODEL

docker build --build-arg LATEST_MODEL --tag test_img -f Dockerfile_deploy .

docker run -d -p 8080:8080 --name=local_imdb test_img

