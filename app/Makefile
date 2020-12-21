build:
	rm -rf ./dist && mkdir ./dist
	cp ./main.py ./dist
	cd ./src && zip -r ../dist/src.zip .

preprocess_train:
	spark-submit --py-files dist/src.zip main.py --job preprocess_data --mode local --phase train

preprocess_train_aws:
	spark-submit --py-files dist/src.zip main.py --job preprocess_data --mode aws --phase train \
	--awsKey ${AWS_KEY} --awsSecretKey ${AWS_SECRET_KEY} --s3Bucket ${S3_BUCKET}

train_model:
	spark-submit --py-files dist/src.zip main.py --job train_model --mode local --phase train

train_model_aws:
	spark-submit --py-files dist/src.zip main.py --job preprocess_data --mode aws --phase train \
	--awsKey ${AWS_KEY} --awsSecretKey ${AWS_SECRET_KEY} --s3Bucket ${S3_BUCKET}

test:
	python -m pytest  --cov src/tests/transformers