.PHONY: setup

help:
	@echo "  setup         setup basic project files"

setup:
	virtualenv -p python3 env && \
	. env/bin/activate && \
	pip install -U pip setuptools wheel && \
	pip install -r requirements.txt && \
	ln -s /mnt/nfs/kaggle/cervix/data/ data && \
	mkdir models
