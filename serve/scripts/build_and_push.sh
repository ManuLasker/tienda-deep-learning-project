#!/bin/bash

echo "Loging to ECR"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Building docker container in $PWD context"
docker build --rm -t $REPOSITORY_NAME .
docker tag $REPOSITORY_NAME:latest $IMAGE_URI:latest

echo "Pushing container to repository $IMAGE_URI:latest"
docker push $IMAGE_URI:latest