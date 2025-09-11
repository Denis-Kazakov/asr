FROM python:3.12

WORKDIR /project

COPY ./nlp/transcript_segmentation/stanza/requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


COPY ./__init__.py ./app/
COPY ./shared ./app/shared
COPY ./nlp/transcript_segmentation/segmenter_base.py ./app/nlp/transcript_segmentation/
COPY ./nlp/transcript_segmentation/routes.py ./app/nlp/transcript_segmentation/
COPY ./nlp/transcript_segmentation/__init__.py ./app/nlp/transcript_segmentation/
COPY ./nlp/transcript_segmentation/stanza ./app/nlp/transcript_segmentation/stanza/

EXPOSE 3002

CMD ["uvicorn", "app.nlp.transcript_segmentation.stanza.src.main:app", "--host", "0.0.0.0", "--port", "3002"]
