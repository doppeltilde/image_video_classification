FROM python:3.12.7

WORKDIR /app

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && \
    apt full-upgrade -y && \
    apt install python3 python3-dev git build-essential cmake libsqlite3-0 libsqlite3-dev -y && \
    pip3 install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Install OpenCL backend for PyTorch
RUN wget https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl && \
    pip3 install pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl && \
    rm pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl

COPY . /app

CMD ["fastapi", "run", "main.py", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]