from fastapi import FastAPI
from app.config.variables import DEBUG

# Crate app
app = FastAPI(debug=DEBUG,
            title="Tienda App Inference Server",
            description="Tienda Inference Server for detection products and classification")

from app import routes