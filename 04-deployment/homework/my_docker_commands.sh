docker build -f my_homework.Dockerfile -t batch_deployment:1.0 .

docker run -it -v /home/avillalbacantero/Documents/Career/Self-Training/MLOps_Zoomcamp/mlops-zoomcamp/data:/app/data -v /home/avillalbacantero/Documents/Career/Self-Training/MLOps_Zoomcamp/mlops-zoomcamp/04-deployment/homework/output:/app/output batch_deployment:1.0