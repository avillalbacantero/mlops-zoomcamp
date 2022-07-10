FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

# Install Python dependencies
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

# Create input and output paths, to map them with local dirs
RUN mkdir "data"
RUN mkdir "output"

# Copy script to run batch inference
COPY ["starter.py", "starter.py"]

# Finally, install dependencies using pipenv
COPY ["Pipfile", "Pipfile"]
COPY ["Pipfile.lock", "Pipfile.lock"]
RUN pipenv install --system --deploy

#ENTRYPOINT [ "python", "starter.py", "--year", "2021", "--month", "4" ]
ENTRYPOINT [ "python", "starter.py" ]
