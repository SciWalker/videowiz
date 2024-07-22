# Use Python 3.12 slim as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install some useful system utilities (optional)
RUN apt-get update && apt-get install -y \
    vim \
    git \
    espeak \
    ffmpeg \

    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run when starting the container
CMD ["/bin/bash"]