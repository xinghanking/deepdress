from fastapi import FastAPI
from app.api import get_age

app = FastAPI()
app.include_router(get_age.router)