# Use the official Python image for 3.11
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application's code into the container
COPY credit-scoring/ .

# Expose default streamlit door
EXPOSE 8501

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory for api
WORKDIR /app/dashboard