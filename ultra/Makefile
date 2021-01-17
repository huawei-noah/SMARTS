# Name of image
IMAGE = ultra
# Docker TAG
TAG = gpu
build:
	docker build \
		-t $(IMAGE):$(TAG) \
		--network=host \
		.
.PHONY: build
