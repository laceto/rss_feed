FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir kitai-0.1.0-py3-none-any.whl

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "output/chatbot_rag.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
