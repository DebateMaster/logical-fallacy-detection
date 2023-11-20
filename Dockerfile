# Use Python 3.10 image as the base
FROM python:3.10-slim-buster

# Set working directory in the container
WORKDIR /app

# Copy only the requirements.txt initially for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 12023

# Command to run the application
CMD ["waitress-serve", "--port=12023", "app:app"]