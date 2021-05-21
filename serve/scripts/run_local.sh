#!/bin/bash
echo "Testing Endpoint Locally"

rebuild=${1:-"rebuild"}
default="rebuild"

if [[ "$rebuild" = "$default" ]]; then
    chmod +x scripts/*.sh
    chmod +x src/serve
    docker build --rm -t tienda-inference-server . -f Dockerfile
fi

docker run --rm -p 8080:8080 -v $(pwd)/models:/opt/ml/models -it tienda-inference-server bash