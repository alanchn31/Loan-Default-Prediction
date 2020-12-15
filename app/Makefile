build:
	rm -rf ./dist && mkdir ./dist
	cp ./main.py ./dist
	cp ./config.json ./dist
	zip -r dist/jobs.zip jobs

preprocess_train:
	spark-submit --py-files dist/jobs.zip --files config.json main.py --job preprocess_data --phase train

test:
	python -m pytest  --cov tests/transformers