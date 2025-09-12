# Base image
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SCAPY_USE_LIBPCAP=no

# Workdir
WORKDIR /app

# Install Python dependencies first (leverages Docker layer cache)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source (entire app folder contents)
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Default command (main entry script in this directory)
CMD ["streamlit", "run", "00_Home.py", "--server.port=8501"]
