import { AnimatePresence } from 'framer-motion';
import PostCard from './PostCard';

export default function Feed({ posts, loading, error, searchTerm, onDelete }) {
  const filtered = searchTerm
    ? posts.filter((p) => p.content.toLowerCase().includes(searchTerm.toLowerCase()))
    : posts;

  if (loading) {
    return <EmptyState title="Loading feed…" subtitle="Fetching posts from the backend." />;
  }

  if (error) {
    return (
      <EmptyState
        title="Couldn't load the feed"
        subtitle="The backend could not reach PostgreSQL. Check DATABASE_URL and make sure the database is running."
      />
    );
  }

  if (filtered.length === 0) {
    return (
      <EmptyState
        title={searchTerm ? 'No posts match your search' : 'No posts yet'}
        subtitle={searchTerm ? 'Try a different keyword.' : 'Your first post will show up here.'}
      />
    );
  }

  return (
    <div>
      <AnimatePresence initial={false}>
        {filtered.map((post) => (
          <PostCard key={post.id} post={post} onDelete={onDelete} />
        ))}
      </AnimatePresence>
    </div>
  );
}

function EmptyState({ title, subtitle }) {
  return (
    <div style={{ padding: '48px 24px', textAlign: 'center' }}>
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.1rem' }}>
        {title}
      </div>
      <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: 6 }}>
        {subtitle}
      </div>
    </div>
  );
}