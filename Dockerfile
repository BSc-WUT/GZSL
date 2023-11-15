FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
COPY packetbaseml/ /app/packetbaseml

RUN pip install --no-cache-dir --upgrade --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
RUN pip install --no-cache-dir --upgrade --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -e ./packetbaseml

EXPOSE 8080

CMD ["uvicorn", "packetbaseml.api.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]
