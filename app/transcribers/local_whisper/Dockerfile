FROM python:3.11

WORKDIR /project

COPY ./transcribers/local_whisper/requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY ./__init__.py ./app/
COPY ./shared ./app/shared
COPY ./transcribers/transcriber_base.py ./app/transcribers/
COPY ./transcribers/routes.py ./app/transcribers/
COPY ./transcribers/__init__.py ./app/transcribers/
COPY ./transcribers/local_whisper ./app/transcribers/local_whisper/

EXPOSE 3001

CMD ["uvicorn", "app.transcribers.local_whisper.src.main:app", "--host", "0.0.0.0", "--port", "3001"]
