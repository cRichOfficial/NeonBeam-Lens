FROM python:3.11-slim

# Install OpenCV backend deps for camera management
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt* ./
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
