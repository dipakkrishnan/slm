.PHONY:	setup install

install:
	pip install -r requirements.txt

setup:
	python3 -m training.scripts.fetch_datasets
	python3 -m training.scripts.train_tokenizer
