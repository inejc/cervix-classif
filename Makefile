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
	nvidia-docker-compose build

run:
	nvidia-docker-compose up -d

stop:
	nvidia-docker-compose stop

r:
	nvidia-docker-compose run head $(cmd)

cli:
	@$(MAKE) r cmd="/bin/bash; cd $UHOME"
