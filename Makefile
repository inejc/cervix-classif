.PHONY: setup
UHOME = /home/user

help:
	@echo "  setup         setup basic project files"

setup-host:
	virtualenv -p python3 env && \
	    . env/bin/activate && \
	    pip install -U pip setuptools wheel && \
	    pip install -r requirements.txt && \
	    mkdir -p models && \
		wget -P models https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

setup-docker:
	sed -i "s/head/$(USER)/g" docker-compose.yml
	nvidia-docker-compose build

run:
	nvidia-docker-compose up -d

stop:
	nvidia-docker-compose stop

r:
	nvidia-docker-compose run $(USER) $(cmd)

cli:
	@$(MAKE) r cmd="/bin/bash; cd $UHOME"
