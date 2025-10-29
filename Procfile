# Force rebuild with openai==1.54.0 to fix proxies error
web: gunicorn esg_app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
