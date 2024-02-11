FROM python:3.11-slim
# RUN apk update
# RUN apk add py-pip
# RUN apk add --no-cache python3-dev 
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app"]