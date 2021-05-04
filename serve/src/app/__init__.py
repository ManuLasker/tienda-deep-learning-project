from fastapi import FastAPI
from app.config.variables import DEBUG

app = FastAPI(debug=DEBUG)

from app import routes
