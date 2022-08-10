local_stack:
	docker-compose up -d

test: local_stack
	pytest tests/

quality_checks: test
	isort .
	black .
	pylint --recursive=y .