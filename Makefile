# Makefile setup mimics: https://github.com/yijunyu/llm.rs/tree/master

.PHONY:	setup install preprocess train run

install:
	pip3 install -r requirements.txt

preprocess:
	python3 preprocessing/preprocess.py

train:	
	cargo build --release && cp target/release/slm ./slm
	./slm

setup:
	install preprocess

all:
	setup train

clean:
	rm -f train
