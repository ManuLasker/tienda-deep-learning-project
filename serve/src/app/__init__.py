from fastapi import FastAPI
from app.config.variables import DEBUG
# from app.dl_model.models import BBPredictor, ClassifierPredictor

# # load predictors before creating the app
# BBPredictor.load_model()
# ClassifierPredictor.load_model()

# Crate app
app = FastAPI(debug=DEBUG)

from app import routes
