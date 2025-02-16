# Use the official Python 3.12 slim-bookworm image as the base
FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files, enable unbuffered logging, and signal we're running in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUNNING_IN_DOCKER=true

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (tesseract, nodejs, and npm)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        nodejs \
        npm \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements files to leverage Docker layer caching
COPY requirements.txt requirements.in /app/

# Install Python dependencies globally
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Create the /data directory required by the project
RUN mkdir -p /data

# Create a non-root user for improved security and adjust directory ownership
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app /data

# Switch to the non-root user
USER app

# Set HOME to a writable directory for the non-root user
ENV HOME=/app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
