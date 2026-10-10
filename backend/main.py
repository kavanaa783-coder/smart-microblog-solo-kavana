# ============================================
# main.py
# Smart Microblog Privacy Guard
# PostgreSQL Version
# ============================================

import logging
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from backend.config import settings
    from backend.database import (
        create_user,
        delete_bookmark_from_db,
        delete_post_from_db,
        get_all_posts,
        get_bookmarks_for_user,
        get_message_conversations,
        get_messages_between_users,
        get_post,
        get_replies_for_post,
        get_statistics,
        get_user,
        init_db,
        rename_user,
        save_bookmark_to_db,
        save_message_to_db,
        save_post_to_db,
        save_reply_to_db,
        toggle_post_like,
        toggle_post_repost,
        update_post,
        update_user,
    )
except ImportError:
    from config import settings
    from database import (
        create_user,
        delete_bookmark_from_db,
        delete_post_from_db,
        get_all_posts,
        get_bookmarks_for_user,
        get_message_conversations,
        get_messages_between_users,
        get_post,
        get_replies_for_post,
        get_statistics,
        get_user,
        init_db,
        save_bookmark_to_db,
        save_message_to_db,
        save_post_to_db,
        save_reply_to_db,
        rename_user,
        toggle_post_like,
        toggle_post_repost,
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
    current_username: str | None = None
    bio: str = ""
    profile_image: str = ""


class BookmarkRequest(BaseModel):
    username: str
    post_id: int


class PostActionRequest(BaseModel):
    username: str


class MessageRequest(BaseModel):
    username: str
    recipient: str
    content: str


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
    username = request.username.strip().removeprefix("@").strip()
    current_username = (request.current_username or request.username).strip().removeprefix("@").strip()

    if not re.fullmatch(r"[A-Za-z0-9_]{1,30}", username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 1–30 characters and contain only letters, numbers, or underscores.",
        )
    if len(request.bio) > 280:
        raise HTTPException(status_code=400, detail="Bio must be 280 characters or fewer.")

    try:
        user = get_user(current_username)
        target_user = get_user(username)

        if current_username != username and target_user is not None:
            raise HTTPException(status_code=409, detail="That username is already taken.")

        if user is not None:
            if current_username == username:
                update_user(username, request.bio, request.profile_image)
            else:
                rename_user(current_username, username, request.bio, request.profile_image)
        elif target_user is None:
            create_user(username, request.bio, request.profile_image)
        else:
            raise HTTPException(status_code=409, detail="That username is already taken.")

        updated_user = get_user(username)
        return updated_user or {
            "username": username,
            "bio": request.bio,
            "profile_image": request.profile_image,
        }
    except HTTPException:
        raise
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
def get_feed(username: str = ""):
    try:
        return {"posts": get_all_posts(username)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch feed")
        raise HTTPException(status_code=503, detail="Unable to load the feed right now.") from exc


@app.post("/post/{post_id}/like")
def toggle_like(post_id: int, request: PostActionRequest):
    try:
        if get_post(post_id) is None:
            raise HTTPException(status_code=404, detail="Post not found")
        user = get_user(request.username)
        user_id = user["id"] if user else create_user(request.username)
        result = toggle_post_like(user_id, post_id)
        return {"liked": result["active"], "like_count": result["count"]}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/post/{post_id}/repost")
def toggle_repost(post_id: int, request: PostActionRequest):
    try:
        if get_post(post_id) is None:
            raise HTTPException(status_code=404, detail="Post not found")
        user = get_user(request.username)
        user_id = user["id"] if user else create_user(request.username)
        result = toggle_post_repost(user_id, post_id)
        return {"reposted": result["active"], "repost_count": result["count"]}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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


# ============================================
# BOOKMARKS
# ============================================

@app.post("/bookmarks")
def save_bookmark(request: BookmarkRequest):
    try:
        user = get_user(request.username)
        if user is None:
            user_id = create_user(request.username)
        else:
            user_id = user["id"]

        save_bookmark_to_db(user_id, request.post_id)
        return {
            "bookmarked": True,
            "post_id": request.post_id,
            "username": request.username,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to save bookmark")
        raise HTTPException(status_code=503, detail="Unable to save the bookmark right now.") from exc


@app.get("/bookmarks/{username}")
def read_bookmarks(username: str):
    try:
        return {"bookmarks": get_bookmarks_for_user(username)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch bookmarks")
        raise HTTPException(status_code=503, detail="Unable to load bookmarks right now.") from exc


@app.delete("/bookmarks/{username}/{post_id}")
def remove_bookmark(username: str, post_id: int):
    try:
        user = get_user(username)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        delete_bookmark_from_db(user["id"], post_id)
        return {"bookmarked": False, "post_id": post_id, "username": username}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to remove bookmark")
        raise HTTPException(status_code=503, detail="Unable to remove the bookmark right now.") from exc


# ============================================
# DIRECT MESSAGES
# ============================================

@app.get("/messages/{username}")
def read_message_conversations(username: str):
    try:
        return {"conversations": get_message_conversations(username)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/messages/{username}/{other_username}")
def read_messages(username: str, other_username: str):
    if username == other_username:
        raise HTTPException(status_code=400, detail="You cannot message yourself.")

    try:
        return {"messages": get_messages_between_users(username, other_username)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/messages")
def send_message(request: MessageRequest):
    username = request.username.strip()
    recipient = request.recipient.strip()
    content = request.content.strip()

    if not username or not recipient:
        raise HTTPException(status_code=400, detail="Sender and recipient are required.")
    if username == recipient:
        raise HTTPException(status_code=400, detail="You cannot message yourself.")
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(content) > 2000:
        raise HTTPException(status_code=400, detail="Messages must be 2000 characters or fewer.")

    try:
        sender = get_user(username)
        sender_id = sender["id"] if sender else create_user(username)
        recipient_user = get_user(recipient)
        recipient_id = recipient_user["id"] if recipient_user else create_user(recipient)
        message = save_message_to_db(sender_id, recipient_id, content)
        return {
            "id": message["id"],
            "sender": username,
            "recipient": recipient,
            "content": content,
            "created_at": message["created_at"],
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to send message")
        raise HTTPException(status_code=503, detail="Unable to send the message right now.") from exc