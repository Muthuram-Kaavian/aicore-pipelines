# # Specify which base layers (default dependencies) to use
# # You may find more base layers at https://hub.docker.com/
# FROM python:3.7
# #
# # Creates directory within your Docker image
# RUN mkdir -p /app/src/
# #
# # Copies file from your Local system TO path in Docker image
# COPY main.py /app/src/
# COPY requirements.txt /app/src/
# #
# # Installs dependencies within you Docker image
# RUN pip3 install -r /app/src/requirements.txt
# #
# # Enable permission to execute anything inside the folder app
# RUN chgrp -R 65534 /app && \
#     chmod -R 777 /app


FROM python:3.10-slim

WORKDIR /app

COPY inference/ ./inference/
WORKDIR /app/inference

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "main.py"]
