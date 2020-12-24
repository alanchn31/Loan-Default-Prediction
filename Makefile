setup:
	virtualenv venv
	source venv/bin/activate 		# In windows, use venv\Scripts\activate
	pip install -r requirements.txt

setup_local_docker:
	docker-compose -f docker-compose.yml up
	