FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY notebook.ipynb .
COPY predictions_visualized.png .

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", "--NotebookApp.token=''"]
