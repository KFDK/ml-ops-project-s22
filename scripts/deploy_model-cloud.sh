cd .. 
export LATEST_MODEL=$(ls -t models/ | head -1)

echo $LATEST_MODEL
IMG_NAME=test_img
echo $IMG_NAME
IMG_TAG=gcr.io/better-mldtu/mlops_pytorch_custom_container:$IMG_NAME
docker build --build-arg LATEST_MODEL --tag $IMG_TAG -f Dockerfile_deploy .

docker push $IMG_TAG
#docker run -d -p 8080:8080 --name=local_imdb test_img

