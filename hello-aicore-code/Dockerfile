# Use official Python image
FROM python:3.7-slim

# Set working directory inside the container
WORKDIR /app/src/

# Copy model, code, and requirements to the container
COPY main.py .
COPY model.joblib .
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Give full permissions (good for SAP AI Core environments)
RUN chgrp -R 65534 /app && chmod -R 777 /app

# Run the main.py script with input when container is triggered
ENTRYPOINT ["python", "main.py"]
