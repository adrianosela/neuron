all: dep run

run:
	./train.py house_prices.csv 10 0.3

dep:
	pip install tensorflow pandas sklearn
