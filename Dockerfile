# Use the official Python image for 3.11
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Command to start the API using uvicorn
CMD ["python", "-m", "uvicorn", "api.main:app", "--reload"]
