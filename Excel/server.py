
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Scan & Discover server is running!"
    }
