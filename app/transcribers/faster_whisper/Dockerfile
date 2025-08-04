FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Top-level dir of the project
WORKDIR /app

COPY ./shared ./shared
COPY ./transcribers/transcriber_base.py ./transcribers/.
COPY ./transcribers/routes.py ./transcribers/.
COPY ./transcribers/faster_whisper ./transcribers/.

# Switch to the service's directory
WORKDIR app/transcribers/faster_whisper

RUN apt-get update -y && apt-get install -y python3-pip
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 3001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3001"]
