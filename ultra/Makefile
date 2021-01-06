# Name of image
IMAGE = ultra
# Docker TAG
TAG = gpu
build:
	docker build \
		-t $(IMAGE):$(TAG) \
		--network=host \
		--build-arg http_proxy=http://localhost:3128 \
		--build-arg https_proxy=https://localhost:3128 \
		--build-arg no_proxy \
		.
.PHONY: build
