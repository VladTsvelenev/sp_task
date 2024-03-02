# syntax=docker/dockerfile:1

# Use the specified Python version as the base image
FROM python:3.10.9-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the dependency files to leverage Docker caching
COPY pyproject.toml poetry.lock /app/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi


RUN mkdir /app/upload
COPY web.py /app/
COPY test.pt /app/
COPY templates /app/templates

# Copy the source code into the container.

# Expose port 5000
EXPOSE 5000
# Run the application using gunicorn
CMD gunicorn 'web:app' --bind=0.0.0.0:5000
