from app import app

@app.get('/ping')
def ping():
    return {'message':'all is well'}

@app.post('/invocation')
def invocation():
    return {'message': 'Not implemented yet!'}