# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Run the simulation script by default
CMD ["python", "src/black_hole_simulation.py", "--config", "configs/simulation_config.yaml"]