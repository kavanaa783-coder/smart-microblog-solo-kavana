import { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import PostCard from '../components/PostCard';
import { getBookmarks, getCurrentUsername } from '../api/client';

export default function Bookmarks() {
  useOutletContext();
  const [bookmarks, setBookmarks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const username = getCurrentUsername();

    getBookmarks(username)
      .then((saved) => setBookmarks(saved))
      .catch(() => setBookmarks([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="main-col">
      <div className="page-header">
        <h1>Bookmarks</h1>
        <p>Your saved posts</p>
      </div>

      <div className="page-body">
        {loading ? (
          <div style={{ padding: 32, color: 'var(--text-tertiary)' }}>Loading bookmarks…</div>
        ) : bookmarks.length === 0 ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-tertiary)' }}>
            No bookmarks yet. Save a post to keep it here.
          </div>
        ) : (
          <div style={{ paddingTop: 12 }}>
            {bookmarks.map((post) => (
              <PostCard key={post.id} post={post} onDelete={() => {}} />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
