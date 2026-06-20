# 1. Use an official Python runtime
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR app

# 3. Create the non-root user 
RUN addgroup --system app && \
    adduser --system --ingroup app --home /home/app app

# Install curl
RUN apt-get update && apt-get install -y curl

# 4. Copy requirements and INSTALL as ROOT
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the app code and change ownership to 'app'
COPY --chown=app:app backend/app /app/app
COPY --chown=app:app kb /app/kb

RUN mkdir -p /app/chroma_data && \
    chown -R app:app /app/chroma_data

# 6. NOW switch to the non-root user for security
USER app

# 7. Expose the port
EXPOSE 8000

# 8. Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
