import psycopg2

try:
    from backend.config import settings
except ImportError:
    from config import settings


# ==========================================
# PostgreSQL Connection
# ==========================================

def get_connection():
    database_url = settings.DATABASE_URL

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Copy .env.example to .env and set DATABASE_URL before starting the API."
        )

    try:
        return psycopg2.connect(database_url)
    except Exception as exc:
        raise RuntimeError(
            "Unable to connect to the configured PostgreSQL database. "
            "Check DATABASE_URL and the database server."
        ) from exc


# ==========================================
# Initialize Database
# ==========================================

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                bio TEXT DEFAULT '',
                profile_image TEXT DEFAULT '',
                joined_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Posts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS posts(
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                risk_level VARCHAR(20),
                risk_score INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Replies table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS replies(
                id SERIAL PRIMARY KEY,
                post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Bookmarks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks(
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, post_id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS post_likes(
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, post_id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS reposts(
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, post_id)
            );
        """)

        # Direct messages
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages(
                id SERIAL PRIMARY KEY,
                sender_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                recipient_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (sender_id <> recipient_id)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS messages_conversation_idx
            ON messages(sender_id, recipient_id, created_at);
        """)

        conn.commit()

    except Exception as exc:
        conn.rollback()
        raise RuntimeError(
            "Database initialization failed. "
            "Check your DATABASE_URL and the PostgreSQL server status."
        ) from exc

    finally:
        cur.close()
        conn.close()


# ==========================================
# Save Post
# ==========================================

def save_post_to_db(user_id, content, risk_level, risk_score):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO posts(user_id, content, risk_level, risk_score)
            VALUES (%s, %s, %s, %s)
            RETURNING id, timestamp
            """,
            (user_id, content, risk_level, risk_score),
        )

        row = cur.fetchone()
        conn.commit()

    except Exception as exc:
        conn.rollback()
        raise RuntimeError(
            "Unable to save the post because the database connection is not available."
        ) from exc

    finally:
        cur.close()
        conn.close()

    if row is None:
        raise RuntimeError("The post could not be created in the database.")

    return {
        "id": row[0],
        "timestamp": row[1],
    }


# ==========================================
# Get All Posts
# ==========================================

def get_all_posts(username=None):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                posts.id,
                users.username,
                posts.content,
                posts.risk_level,
                posts.risk_score,
                posts.timestamp,
                (SELECT COUNT(*) FROM post_likes WHERE post_likes.post_id = posts.id),
                EXISTS (
                    SELECT 1 FROM post_likes
                    JOIN users ON users.id = post_likes.user_id
                    WHERE post_likes.post_id = posts.id AND users.username = %s
                ),
                (SELECT COUNT(*) FROM reposts WHERE reposts.post_id = posts.id),
                EXISTS (
                    SELECT 1 FROM reposts
                    JOIN users ON users.id = reposts.user_id
                    WHERE reposts.post_id = posts.id AND users.username = %s
                )
            FROM posts
            JOIN users
                ON posts.user_id = users.id
            ORDER BY posts.id DESC
        """, (username or "", username or ""))

        rows = cur.fetchall()

    finally:
        cur.close()
        conn.close()

    posts = []

    for row in rows:
        posts.append({
            "id": row[0],
            "username": row[1],
            "content": row[2],
            "risk_level": row[3],
            "risk_score": row[4],
            "timestamp": row[5],
            "like_count": row[6],
            "liked_by_user": row[7],
            "repost_count": row[8],
            "reposted_by_user": row[9],
        })

    return posts


# ==========================================
# Get Single Post
# ==========================================

def get_post(post_id):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                posts.id,
                users.username,
                posts.content,
                posts.risk_level,
                posts.risk_score,
                posts.timestamp
            FROM posts
            JOIN users
                ON posts.user_id = users.id
            WHERE posts.id = %s
        """, (post_id,))

        row = cur.fetchone()

    finally:
        cur.close()
        conn.close()

    if row:
        return {
            "id": row[0],
            "username": row[1],
            "content": row[2],
            "risk_level": row[3],
            "risk_score": row[4],
            "timestamp": row[5],
        }

    return None


# ==========================================
# Update Post
# ==========================================

def update_post(post_id, content):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            UPDATE posts
            SET content = %s
            WHERE id = %s
        """, (content, post_id))

        conn.commit()

    finally:
        cur.close()
        conn.close()


# ==========================================
# Delete Post
# ==========================================

def delete_post_from_db(post_id):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            DELETE FROM posts
            WHERE id = %s
        """, (post_id,))

        conn.commit()

    finally:
        cur.close()
        conn.close()


# ==========================================
# Save Reply
# ==========================================

def save_reply_to_db(post_id, user_id, content):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO replies(post_id, user_id, content)
            VALUES (%s, %s, %s)
            RETURNING id, timestamp
            """,
            (post_id, user_id, content),
        )

        row = cur.fetchone()
        conn.commit()

    except Exception as exc:
        conn.rollback()
        raise RuntimeError(
            "Unable to save the reply because the database connection is not available."
        ) from exc

    finally:
        cur.close()
        conn.close()

    if row is None:
        raise RuntimeError("The reply could not be created.")

    return {
        "id": row[0],
        "timestamp": row[1],
    }


# ==========================================
# Bookmarks
# ==========================================

def save_bookmark_to_db(user_id, post_id):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO bookmarks(user_id, post_id)
            VALUES (%s, %s)
            ON CONFLICT (user_id, post_id) DO NOTHING
            RETURNING id, created_at
            """,
            (user_id, post_id),
        )
        row = cur.fetchone()
        conn.commit()

    except Exception as exc:
        conn.rollback()
        raise RuntimeError(
            "Unable to save the bookmark because the database connection is not available."
        ) from exc

    finally:
        cur.close()
        conn.close()

    if row is None:
        return {"id": None, "created_at": None}

    return {
        "id": row[0],
        "created_at": row[1],
    }


def get_bookmarks_for_user(username):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT
                bookmarks.id,
                bookmarks.post_id,
                users.username,
                posts.content,
                posts.risk_level,
                posts.risk_score,
                posts.timestamp,
                bookmarks.created_at
            FROM bookmarks
            JOIN users ON bookmarks.user_id = users.id
            JOIN posts ON bookmarks.post_id = posts.id
            WHERE users.username = %s
            ORDER BY bookmarks.created_at DESC
            """,
            (username,),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    bookmarks = []
    for row in rows:
        bookmarks.append({
            "id": row[0],
            "post_id": row[1],
            "username": row[2],
            "content": row[3],
            "risk_level": row[4],
            "risk_score": row[5],
            "timestamp": row[6],
            "created_at": row[7],
        })

    return bookmarks


def delete_bookmark_from_db(user_id, post_id):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            DELETE FROM bookmarks
            WHERE user_id = %s AND post_id = %s
            """,
            (user_id, post_id),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def toggle_post_like(user_id, post_id):
    return _toggle_post_action("post_likes", user_id, post_id)


def toggle_post_repost(user_id, post_id):
    return _toggle_post_action("reposts", user_id, post_id)


def _toggle_post_action(table, user_id, post_id):
    if table not in {"post_likes", "reposts"}:
        raise ValueError("Unsupported post action table")

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            f"SELECT 1 FROM {table} WHERE user_id = %s AND post_id = %s",
            (user_id, post_id),
        )
        exists = cur.fetchone() is not None

        if exists:
            cur.execute(
                f"DELETE FROM {table} WHERE user_id = %s AND post_id = %s",
                (user_id, post_id),
            )
        else:
            cur.execute(
                f"INSERT INTO {table}(user_id, post_id) VALUES (%s, %s)",
                (user_id, post_id),
            )

        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE post_id = %s", (post_id,))
        count = cur.fetchone()[0]
        conn.commit()
        return {"active": not exists, "count": count}
    except Exception as exc:
        conn.rollback()
        raise RuntimeError("Unable to update this post action right now.") from exc
    finally:
        cur.close()
        conn.close()


# ==========================================
# Get Replies For Post
# ==========================================

def get_replies_for_post(post_id):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                replies.id,
                replies.post_id,
                users.username,
                replies.content,
                replies.timestamp
            FROM replies
            JOIN users
                ON replies.user_id = users.id
            WHERE replies.post_id = %s
            ORDER BY replies.id ASC
        """, (post_id,))

        rows = cur.fetchall()

    finally:
        cur.close()
        conn.close()

    replies = []

    for row in rows:
        replies.append({
            "id": row[0],
            "post_id": row[1],
            "username": row[2],
            "content": row[3],
            "timestamp": row[4],
        })

    return replies


# ==========================================
# Statistics
# ==========================================

def get_statistics():
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM posts")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM posts WHERE risk_level = 'LOW'")
        low = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM posts WHERE risk_level = 'MEDIUM'")
        medium = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM posts WHERE risk_level = 'HIGH'")
        high = cur.fetchone()[0]

    finally:
        cur.close()
        conn.close()

    return {
        "total_posts": total,
        "low_risk": low,
        "medium_risk": medium,
        "high_risk": high,
    }


# ==========================================
# USER FUNCTIONS
# ==========================================

def create_user(username, bio="", profile_image=""):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO users(username, bio, profile_image)
            VALUES(%s, %s, %s)
            RETURNING id
        """, (username, bio, profile_image))

        user_id = cur.fetchone()[0]
        conn.commit()

    finally:
        cur.close()
        conn.close()

    return user_id


def get_user(username):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT id, username, bio, profile_image
            FROM users
            WHERE username = %s
        """, (username,))

        row = cur.fetchone()

    finally:
        cur.close()
        conn.close()

    if row:
        return {
            "id": row[0],
            "username": row[1],
            "bio": row[2],
            "profile_image": row[3],
        }

    return None


def update_user(username, bio, profile_image=""):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            UPDATE users
            SET bio = %s, profile_image = %s
            WHERE username = %s
        """, (bio, profile_image, username))

        conn.commit()

    finally:
        cur.close()
        conn.close()


def rename_user(current_username, username, bio, profile_image=""):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            UPDATE users
            SET username = %s, bio = %s, profile_image = %s
            WHERE username = %s
            RETURNING id
            """,
            (username, bio, profile_image, current_username),
        )
        row = cur.fetchone()
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise RuntimeError("Unable to update the username right now.") from exc
    finally:
        cur.close()
        conn.close()

    if row is None:
        raise RuntimeError("The current profile could not be found.")

    return row[0]


# ==========================================
# Direct Messages
# ==========================================

def save_message_to_db(sender_id, recipient_id, content):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO messages(sender_id, recipient_id, content)
            VALUES (%s, %s, %s)
            RETURNING id, created_at
            """,
            (sender_id, recipient_id, content),
        )
        row = cur.fetchone()
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise RuntimeError("Unable to save the message right now.") from exc
    finally:
        cur.close()
        conn.close()

    return {"id": row[0], "created_at": row[1]}


def get_messages_between_users(username, other_username):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT messages.id, sender.username, recipient.username,
                   messages.content, messages.created_at
            FROM messages
            JOIN users AS sender ON sender.id = messages.sender_id
            JOIN users AS recipient ON recipient.id = messages.recipient_id
            WHERE (sender.username = %s AND recipient.username = %s)
               OR (sender.username = %s AND recipient.username = %s)
            ORDER BY messages.created_at ASC, messages.id ASC
            """,
            (username, other_username, other_username, username),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return [
        {
            "id": row[0],
            "sender": row[1],
            "recipient": row[2],
            "content": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


def get_message_conversations(username):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT username, last_message, updated_at
            FROM (
                SELECT DISTINCT ON (partner.username)
                       partner.username,
                       messages.content AS last_message,
                       messages.created_at AS updated_at
                FROM messages
                JOIN users AS sender ON sender.id = messages.sender_id
                JOIN users AS recipient ON recipient.id = messages.recipient_id
                JOIN users AS partner ON partner.id = CASE
                    WHEN sender.username = %s THEN recipient.id
                    ELSE sender.id
                END
                WHERE sender.username = %s OR recipient.username = %s
                ORDER BY partner.username, messages.created_at DESC, messages.id DESC
            ) AS conversations
            ORDER BY updated_at DESC, username ASC
            """,
            (username, username, username),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return [
        {"username": row[0], "last_message": row[1], "updated_at": row[2]}
        for row in rows
    ]