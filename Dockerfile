ARG VERSION=latest

FROM tiangolo/uvicorn-gunicorn-fastapi:${VERSION}


COPY ./src /app/
COPY ./requirements.txt /app

RUN pip install wheel
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "api:app", "--bind", "0.0.0.0:80"]

