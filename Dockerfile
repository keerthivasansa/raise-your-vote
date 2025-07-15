# Use official Python base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Set environment variables (optional but recommended)
ENV FLASK_APP=final.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the app
CMD ["flask", "run"]
