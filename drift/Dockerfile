FROM python:3.10-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY drift_math.py monitor.py ./
EXPOSE 8002
CMD ["python", "-u", "monitor.py"]