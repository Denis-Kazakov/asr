FROM python:3.12

# Top-level dir of the project
WORKDIR /app

COPY ./shared ./shared
COPY ./transcribers/transcriber_base.py ./transcribers/.
COPY ./transcribers/routes.py ./transcribers/.
COPY ./transcribers/local_whisper ./transcribers/.

# Switch to the service's directory
WORKDIR app/transcribers/local_whisper

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 3001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3001"]
