FROM python:3.7-slim

WORKDIR /app/

COPY main.py .
COPY model.joblib .
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN chgrp -R 65534 /app && chmod -R 777 /app

ENTRYPOINT ["python", "main.py"]
