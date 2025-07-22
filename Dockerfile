# Use official Python image
FROM python:3.13.5-slim


# Set working directory inside container to project root (/app)
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory (all files and folders)
COPY . .

# Expose Jupyter notebook port (optional)
EXPOSE 8888

# Default command to run Jupyter Notebook inside /app
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
