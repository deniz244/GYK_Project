FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv

WORKDIR /gyk_1

COPY gyk_1/. .

RUN python3 -m venv ./venv && \
    . venv/bin/activate && \
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install -r requirements.txt

CMD ["./venv/bin/python", "main.py"]
# CMD ["./venv/bin/uvicorn", "gyk_1.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
