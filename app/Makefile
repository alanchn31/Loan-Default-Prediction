build:
	rm -rf ./dist && mkdir ./dist
	cp ./main.py ./dist
	cp ./config.json ./dist
	cd ./src && zip -r ../dist/src.zip .

# Requires aws cli
package_s3:
	aws s3 --region us-east-1 cp ./dist/src.zip s3://${S3_BUCKET}/dist/src.zip --acl public-read
	aws s3 --region us-east-1 cp ./dist/main.py s3://${S3_BUCKET}/dist/main.py --acl public-read
	aws s3 --region us-east-1 cp ./dist/src.zip s3://${S3_BUCKET}/dist/config.json --acl public-read

preprocess_train:
	spark-submit --py-files dist/src.zip main.py --job preprocess_data --mode local --phase train

preprocess_train_aws:
	spark-submit --py-files dist/src.zip main.py --job preprocess_data --mode aws --phase train \
	--awsKey ${AWS_KEY} --awsSecretKey ${AWS_SECRET_KEY} --s3Bucket ${S3_BUCKET}

train_model:
	spark-submit --py-files dist/src.zip main.py --job train_model --mode local --phase train

train_model_aws:
	spark-submit --py-files dist/src.zip main.py --job train_model --mode aws --phase train \
	--awsKey ${AWS_KEY} --awsSecretKey ${AWS_SECRET_KEY} --s3Bucket ${S3_BUCKET}

preprocess_predict:
	spark-submit --py-files dist/src.zip main.py --job preprocess_data --mode local --phase predict

preprocess_predict_aws:
	spark-submit --py-files dist/src.zip main.py --job preprocess_data --mode aws --phase predict \
	--awsKey ${AWS_KEY} --awsSecretKey ${AWS_SECRET_KEY} --s3Bucket ${S3_BUCKET}

model_predict:
	spark-submit --py-files dist/src.zip main.py --job inference --mode local --phase predict

model_predict_aws:
	spark-submit --py-files dist/src.zip main.py --job inference --mode aws --phase predict \
	--awsKey ${AWS_KEY} --awsSecretKey ${AWS_SECRET_KEY} --s3Bucket ${S3_BUCKET}

test:
	python -m pytest  --cov src/tests/transformers