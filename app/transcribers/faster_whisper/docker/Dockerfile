FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /src

RUN apt-get update -y && apt-get install -y python3-pip

COPY ../src /app/src
COPY ../../transcriber_base.py /app/src/.
COPY ../../../data_models.py /app/src/.
COPY ../../../logging_config.py /app/src/.

RUN pip install faster-whisper
CMD ["python3", "infer.py"]






# Копирование зависимостей
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install -r requirements.txt

# Копирование исходников
COPY ./src /app/src


# Открытие порта
EXPOSE 5851

# Запуск uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5851"]



COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


COPY .env .env

EXPOSE 3001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3001"]
