import { useEffect, useMemo, useState } from 'react';
import { FiHash, FiSearch, FiTrendingUp } from 'react-icons/fi';
import RiskBadge from '../components/RiskBadge';
import { getFeed } from '../api/client';

function findHashtags(posts) {
  const counts = new Map();
  posts.forEach((post) => {
    const tags = post.content.match(/#[\p{L}\p{N}_]+/gu) || [];
    tags.forEach((tag) => {
      const normalized = tag.toLowerCase();
      const current = counts.get(normalized) || { tag, count: 0 };
      current.count += 1;
      counts.set(normalized, current);
    });
  });
  return [...counts.values()].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag)).slice(0, 8);
}

export default function Explore() {
  const [posts, setPosts] = useState([]);
  const [query, setQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    getFeed()
      .then((data) => {
        if (!cancelled) {
          setPosts(data);
          setError('');
        }
      })
      .catch(() => {
        if (!cancelled) setError('Could not load public posts. Check that the API and database are running.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const trends = useMemo(() => findHashtags(posts), [posts]);
  const filteredPosts = useMemo(() => {
    const term = query.trim().toLowerCase();
    return posts.filter((post) => {
      const matchesText = !term || `${post.content} ${post.username}`.toLowerCase().includes(term);
      const matchesTag = !selectedTag || post.content.toLowerCase().includes(selectedTag.toLowerCase());
      return matchesText && matchesTag;
    });
  }, [posts, query, selectedTag]);

  return (
    <main className="main-col">
      <div className="page-header">
        <h1>Explore</h1>
        <p>Search posts shared in the public feed and browse hashtags found in them.</p>
      </div>

      <div className="page-body" style={{ padding: 20, display: 'grid', gap: 18 }}>
        <section className="card" style={{ padding: 18 }}>
          <label htmlFor="explore-search" style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, fontWeight: 700 }}>
            <FiSearch color="var(--brand)" /> Search public posts
          </label>
          <input
            id="explore-search"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search text, usernames, or hashtags"
            style={{ width: '100%', border: '1px solid var(--border)', background: 'var(--surface-alt)', color: 'var(--text-primary)', borderRadius: 12, padding: '12px 14px' }}
          />
          <p style={{ margin: '9px 0 0', color: 'var(--text-tertiary)', fontSize: '0.78rem' }}>Results come from posts currently saved in the app feed; trend counts are calculated from those posts.</p>
        </section>

        <section className="card" style={{ padding: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <FiTrendingUp color="var(--brand)" />
            <strong>Trending hashtags</strong>
            <span style={{ color: 'var(--text-tertiary)', fontSize: '0.76rem' }}>from feed posts</span>
          </div>
          {trends.length ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {trends.map((trend) => (
                <button
                  key={trend.tag.toLowerCase()}
                  type="button"
                  onClick={() => setSelectedTag((current) => current === trend.tag ? '' : trend.tag)}
                  aria-pressed={selectedTag.toLowerCase() === trend.tag.toLowerCase()}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 6, border: '1px solid var(--border)', borderRadius: 999, padding: '8px 12px', cursor: 'pointer', background: selectedTag.toLowerCase() === trend.tag.toLowerCase() ? 'var(--brand)' : 'var(--surface-alt)', color: selectedTag.toLowerCase() === trend.tag.toLowerCase() ? '#fff' : 'var(--text-primary)' }}
                >
                  <FiHash size={13} /> {trend.tag} <span style={{ opacity: 0.75 }}>· {trend.count}</span>
                </button>
              ))}
              {selectedTag ? <button type="button" onClick={() => setSelectedTag('')} style={{ border: 0, background: 'transparent', color: 'var(--brand)', cursor: 'pointer' }}>Clear filter</button> : null}
            </div>
          ) : <p style={{ margin: 0, color: 'var(--text-tertiary)', fontSize: '0.85rem' }}>No hashtags in the feed yet. Hashtags from new posts will appear here.</p>}
        </section>

        <section aria-live="polite" style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
            <h2 style={{ margin: 0, fontSize: '1.05rem' }}>Public posts</h2>
            <span style={{ color: 'var(--text-tertiary)', fontSize: '0.8rem' }}>{filteredPosts.length} {filteredPosts.length === 1 ? 'post' : 'posts'}</span>
          </div>

          {loading ? <div className="card" style={{ padding: 18 }}>Loading posts from the feed…</div> : null}
          {error ? <div role="alert" className="card" style={{ padding: 18, color: 'var(--danger, #d33)' }}>{error}</div> : null}
          {!loading && !error && filteredPosts.length === 0 ? (
            <div className="card" style={{ padding: 20, color: 'var(--text-secondary)' }}>
              {posts.length === 0 ? 'There are no public posts yet. Publish a post on Home and it will appear here.' : 'No posts match this search. Try a different word or clear the hashtag filter.'}
            </div>
          ) : null}

          {filteredPosts.map((post) => (
            <article key={post.id} className="card" style={{ padding: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <strong>{post.username}</strong>
                <span style={{ color: 'var(--text-tertiary)', fontSize: '0.82rem' }}>@{post.username}</span>
                <span style={{ color: 'var(--text-tertiary)' }}>·</span>
                <time style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem' }}>{new Date(post.createdAt).toLocaleString()}</time>
                <span style={{ marginLeft: 'auto' }}><RiskBadge level={post.riskLevel} score={post.riskScore} /></span>
              </div>
              <p style={{ margin: '12px 0 0', lineHeight: 1.55, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{post.content}</p>
            </article>
          ))}
        </section>
      </div>
    </main>
  );
}
