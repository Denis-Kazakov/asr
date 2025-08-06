FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# ENV PYTHONPATH=/app

# Top-level dir of the project
WORKDIR /project

COPY ./transcribers/faster_whisper/requirements.txt .

RUN apt-get update -y && apt-get install -y python3-pip
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY ./__init__.py ./app/
COPY ./shared ./app/shared
COPY ./transcribers/transcriber_base.py ./app/transcribers/
COPY ./transcribers/routes.py ./app/transcribers/
COPY ./transcribers/__init__.py ./app/transcribers/
COPY ./transcribers/faster_whisper ./app/transcribers/faster_whisper/

EXPOSE 3001

CMD ["uvicorn", "app.transcribers.faster_whisper.src.main:app", "--host", "0.0.0.0", "--port", "3001"]
