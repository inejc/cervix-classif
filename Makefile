.PHONY: setup
UHOME = /home/user

help:
	@echo "  setup         setup basic project files"

setup-host:
	virtualenv -p python3 env && \
	    . env/bin/activate && \
	    pip install -U pip setuptools wheel && \
	    pip install -r requirements.txt && \
	    mkdir models

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
