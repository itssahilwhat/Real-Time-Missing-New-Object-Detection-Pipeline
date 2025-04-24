FROM python:3.12-slim

# Install system deps
RUN apt-get update && \
    apt-get install -y libgl1 git && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Entry point
CMD ["python", "main.py"]