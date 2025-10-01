# Imagem base leve (Debian slim para arm64/amd64 — roda bem no Mac M1/M2)
FROM python:3.11-slim-buster

# Diretório de trabalho dentro do container
WORKDIR /app

# Variáveis para comportamento previsível do Python/pip e porta padrão
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80

# Instala dependências do Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o código da aplicação
COPY . .

# Expõe a porta (informativo)
EXPOSE 80

# Comando de inicialização (host 0.0.0.0 para aceitar conexões externas)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--reload", "--port", "8001"]
