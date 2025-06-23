FROM python:3.11

WORKDIR /app

COPY main.py /app/
COPY model.pkl /mnt/models/model.pkl
COPY requirements.txt /app/

RUN pip install -r requirements.txt

EXPOSE 9001
CMD ["python3", "main.py"]
