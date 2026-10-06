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

def get_all_posts():
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
            ORDER BY posts.id DESC
        """)

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