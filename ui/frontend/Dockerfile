FROM python:3.10-slim

WORKDIR /repo

COPY ui/frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/repo

EXPOSE 7860

CMD ["python", "ui/frontend/app.py"]

