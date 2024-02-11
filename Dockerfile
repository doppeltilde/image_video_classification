FROM python:3.11-slim
# RUN apk update
# RUN apk add py-pip
# RUN apk add --no-cache python3-dev 
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]