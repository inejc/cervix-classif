.PHONY: setup

help:
	@echo "  setup         setup basic project files"

setup:
	python3 -m venv env && \
	. env/bin/activate && \
	pip install -U pip setuptools wheel && \
	pip install -r requirements.txt && \
	ln -s /mnt/nfs/kaggle/cervix/data/ data && \
	mkdir models
