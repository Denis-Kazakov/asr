FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update -y && apt-get install -y python3-pip

COPY ../src /app/src
COPY ../../transcriber_base.py /app/src/.
COPY ../../../data_models.py /app/src/.
COPY ../../../logging_config.py /app/src/.

RUN pip install faster-whisper
CMD ["python3", "infer.py"]





# Установка Python
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && python -m pip install --upgrade pip

# Копирование зависимостей
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install -r requirements.txt

# Копирование исходников
COPY ./src /app/src

# Переменные окружения
ENV PYTHONPATH=/app:$PYTHONPATH
ENV FLASK_ENV=dev
ENV FLASK_APP=src/main.py

# Открытие порта
EXPOSE 5851

# Запуск uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5851"]



COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


COPY .env .env

EXPOSE 3001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3001"]
