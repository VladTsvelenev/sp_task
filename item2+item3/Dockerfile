# syntax=docker/dockerfile:1

FROM python:3.10.9-slim as base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install poetry
COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

COPY web.py /app/
COPY test.pt /app/
COPY templates /app/templates

EXPOSE 5000

CMD gunicorn 'web:app' --bind=0.0.0.0:5000
