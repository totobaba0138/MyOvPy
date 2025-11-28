from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
from core import load_global_model, unload_global_model
from routers import stocking

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_global_model()
    yield
    unload_global_model()

app = FastAPI(lifespan=lifespan)

# 挂载丝袜专用路由
app.include_router(stocking.router, prefix="/api", tags=["JAV"])

if __name__ == "__main__":
    # 保持单线程处理
    uvicorn.run("MyOvPy.main:MyOvPy", host="0.0.0.0", port=8000, workers=1)