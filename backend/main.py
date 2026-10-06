# ============================================
# main.py
# Smart Microblog Privacy Guard
# PostgreSQL Version
# ============================================

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from backend.config import settings
    from backend.database import (
    create_user,
    delete_post_from_db,
    get_all_posts,
    get_post,
    get_replies_for_post,
    get_statistics,
    get_user,
    init_db,
    save_post_to_db,
    save_reply_to_db,
    update_post,
    update_user,
)
except ImportError:
    from config import settings
    from database import (
        create_user,
        delete_post_from_db,
        get_all_posts,
        get_post,
        get_statistics,
        get_user,
        init_db,
        save_post_to_db,
        update_post,
        update_user,
    )

try:
    from backend.pii_detector import detect_pii
    from backend.risk_scorer import calculate_risk
except ImportError:
    from pii_detector import detect_pii
    from risk_scorer import calculate_risk

logger = logging.getLogger(__name__)

# ============================================
# FASTAPI
# ============================================

app = FastAPI(
    title="Smart Microblog Privacy Guard",
    description="AI Powered Privacy Detection System",
    version="4.0",
)

# ============================================
# CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    try:
        init_db()
    except RuntimeError as exc:
        logger.warning("Database startup check skipped: %s", exc)


# ============================================
# Request Models
# ============================================

class ScanRequest(BaseModel):
    text: str


class SaveRequest(BaseModel):
    username: str
    content: str
    risk_level: str
    risk_score: int


class UpdateRequest(BaseModel):
    content: str


class ProfileRequest(BaseModel):
    username: str
    bio: str = ""
    profile_image: str = ""


# ============================================
# HOME
# ============================================

@app.get("/")
def home():
    return {"message": "Smart Microblog Privacy Guard API Running"}


@app.get("/health")
def health():
    try:
        try:
            from backend.database import get_connection
        except ImportError:
            from database import get_connection

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return {"status": "ok", "api": "running", "database": "connected"}
    except Exception:
        return {
            "status": "ok",
            "api": "running",
            "database": "unavailable",
            "message": "API is running, but the database is not currently reachable.",
        }


# ============================================
# SCAN POST
# ============================================
@app.post("/scan-post")
def scan_post(request: ScanRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    detected = detect_pii(request.text)
    risk = calculate_risk(detected)

    return {
        "detected_entities": detected,
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "recommendation": risk["recommendation"],
    }


# ============================================
# SAVE POST
# ============================================
@app.post("/save-post")
def save_post(request: SaveRequest):
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Post content cannot be empty.")

    try:
        user = get_user(request.username)

        if user is None:
            user_id = create_user(request.username)
        else:
            user_id = user["id"]

        created = save_post_to_db(
            user_id,
            request.content,
            request.risk_level,
            request.risk_score,
        )

        if created is None or not isinstance(created, dict) or 'id' not in created:
            return {"message": "Post saved successfully"}

        saved_post = get_post(created["id"])
        if saved_post is None:
            return {"message": "Post saved successfully"}
        return saved_post
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to save post")
        raise HTTPException(
            status_code=503,
            detail="Unable to save the post at the moment. Check the database configuration.",
        ) from exc


# ============================================
# SAVE / UPDATE PROFILE
# ============================================
@app.post("/profile")
def save_profile(request: ProfileRequest):
    try:
        user = get_user(request.username)

        if user:
            update_user(
                request.username,
                request.bio,
                request.profile_image,
            )
            updated_user = get_user(request.username)
            return updated_user or {"message": "Profile updated successfully"}

        user_id = create_user(
            request.username,
            request.bio,
            request.profile_image,
        )
        updated_user = get_user(request.username)
        return updated_user or {"id": user_id, "username": request.username, "bio": request.bio, "profile_image": request.profile_image}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to save profile")
        raise HTTPException(
            status_code=503,
            detail="Unable to save the profile at the moment. Check the database configuration.",
        ) from exc


# ============================================
# GET PROFILE
# ============================================

@app.get("/profile/{username}")
def read_profile(username: str):
    try:
        user = get_user(username)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch profile")
        raise HTTPException(
            status_code=503,
            detail="Unable to load the profile right now.",
        ) from exc

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user["id"],
        "username": user["username"],
        "bio": user["bio"],
        "profile_image": user["profile_image"],
    }


# ============================================
# FEED
# ============================================

@app.get("/get-feed")
def get_feed():
    try:
        return {"posts": get_all_posts()}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch feed")
        raise HTTPException(status_code=503, detail="Unable to load the feed right now.") from exc


# ============================================
# GET SINGLE POST
# ============================================

@app.get("/post/{post_id}")
def read_post(post_id: int):
    post = get_post(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return post
# ============================================
# REPLIES
# ============================================

class ReplyRequest(BaseModel):
    username: str
    content: str


@app.post("/post/{post_id}/replies")
def create_reply(post_id: int, request: ReplyRequest):
    if not request.content or not request.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Reply cannot be empty."
        )

    post = get_post(post_id)

    if post is None:
        raise HTTPException(
            status_code=404,
            detail="Post not found."
        )

    try:
        user = get_user(request.username)

        if user is None:
            user_id = create_user(request.username)
        else:
            user_id = user["id"]

        reply = save_reply_to_db(
            post_id,
            user_id,
            request.content.strip(),
        )

        return {
            "id": reply["id"],
            "post_id": post_id,
            "username": request.username,
            "content": request.content.strip(),
            "timestamp": reply["timestamp"],
        }

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.get("/post/{post_id}/replies")
def read_replies(post_id: int):
    post = get_post(post_id)

    if post is None:
        raise HTTPException(
            status_code=404,
            detail="Post not found."
        )

    try:
        return {
            "replies": get_replies_for_post(post_id)
        }
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


# ============================================
# UPDATE POST
# ============================================

@app.put("/update-post/{post_id}")
def edit_post(post_id: int, request: UpdateRequest):
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Post content cannot be empty.")

    update_post(post_id, request.content)
    updated_post = get_post(post_id)
    if updated_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return updated_post


# ============================================
# DELETE POST
# ============================================

@app.delete("/delete-post/{post_id}")
def delete_post(post_id: int):
    try:
        delete_post_from_db(post_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete post")
        raise HTTPException(status_code=503, detail="Unable to delete the post right now.") from exc

    return {"message": "Post deleted successfully"}


# ============================================
# STATISTICS
# ============================================

@app.get("/stats")
def statistics():
    try:
        return get_statistics()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch statistics")
        raise HTTPException(status_code=503, detail="Unable to load statistics right now.") from exc