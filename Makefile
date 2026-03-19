.PHONY: data train predict all clean lint

all: data train predict

data:
	cd data && ./process_data.sh

train:
	cd models && python model.py

predict:
	cd src && for r in 1 2 3 4 5; do ./play_round.sh $$r; done && python predictor.py --round 6

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f models/*.pth

lint:
	python -m flake8 data/ models/ src/ config.py --max-line-length=120 --ignore=E402,W503
