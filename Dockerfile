FROM python:3.13-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ .

EXPOSE 8000

CMD ["python3", "/app/app.py"]