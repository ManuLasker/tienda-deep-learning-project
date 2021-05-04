from app import app
import uvicorn

if __name__ == '__main__':
    uvicorn.run("wsgi:app", reload=True, debug=True,
                port=8080, host='0.0.0.0')