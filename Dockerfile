FROM python:3.13-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./app/ .
CMD ["python3", "/app/app.py"]