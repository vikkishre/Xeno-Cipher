FROM python:3.9-slim

WORKDIR /app

COPY Test/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Test/ .

EXPOSE 5000

# Run the application
CMD ["python", "app.py"] 