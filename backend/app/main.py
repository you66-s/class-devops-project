from fastapi import FastAPI, APIRouter
from .api.routes import router

app = FastAPI()
app.include_router(router=router)

