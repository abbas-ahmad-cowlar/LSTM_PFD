FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY dash_app/requirements.txt dash_app/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r dash_app/requirements.txt

# Copy application code
COPY . .

WORKDIR /app/dash_app

EXPOSE 8050

CMD ["python", "app.py"]
