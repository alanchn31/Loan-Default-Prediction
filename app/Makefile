build:
	rm -rf ./dist && mkdir ./dist
	cp ./main.py ./dist
	cd ./src && zip -r ../dist/src.zip .

preprocess_train:
	spark-submit --py-files dist/src.zip main.py --job preprocess_data --phase train

train_model:
	spark-submit --py-files dist/src.zip main.py --job train_model --phase train

test:
	python -m pytest  --cov src/tests/transformers