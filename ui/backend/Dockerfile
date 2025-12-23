FROM python:3.10-slim

WORKDIR /repo

COPY ui/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Code is volume-mounted by docker-compose for local dev/demo.
ENV PYTHONPATH=/repo

EXPOSE 8000

CMD ["uvicorn", "ui.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

