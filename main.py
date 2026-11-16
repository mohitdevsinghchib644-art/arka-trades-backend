import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

from database import init_supabase
from routers.screener import router as screener_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("arka.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Arka Trades Screener API...")
    init_supabase()
    yield
    logger.info("Shutting down Arka Trades Screener API.")


app = FastAPI(
    title="Arka Trades Screener API",
    description="High-performance decoupled backend for the Arka Trades stock screener.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Configuration
# Allows localhost during local development and FRONTEND_URL in production (Cloudflare Pages)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

frontend_env = os.getenv("FRONTEND_URL", "").strip()
if frontend_env:
    for url in frontend_env.split(","):
        cleaned = url.strip().rstrip("/")
        if cleaned and cleaned not in origins:
            origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.pages\.dev",  # Matches all Cloudflare Pages preview & production subdomains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint for Render service monitoring and keep-awake cron jobs."""
    return {
        "status": "ok",
        "service": "arka-trades-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
    }


# Mount Screener Router
app.include_router(screener_router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
