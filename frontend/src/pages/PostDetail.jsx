import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { FiArrowLeft } from 'react-icons/fi';
import RiskBadge from '../components/RiskBadge';
import { getPost } from '../api/client';

export default function PostDetail() {
  const { postId } = useParams();
  const [post, setPost] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');

    getPost(postId)
      .then((data) => {
        if (active) setPost(data);
      })
      .catch((requestError) => {
        if (active) {
          setError(requestError?.response?.status === 404
            ? 'This post could not be found.'
            : 'Could not load this post. Please try again.');
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [postId]);

  return (
    <main className="main-col">
      <div className="page-header">
        <h1>Shared post</h1>
        <p>Privacy Guard · Post {postId}</p>
      </div>

      <div className="page-body" style={{ padding: 20 }}>
        <Link to="/" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, marginBottom: 16, color: 'var(--brand)', textDecoration: 'none' }}>
          <FiArrowLeft size={16} /> Back to feed
        </Link>

        {loading ? (
          <div className="card" style={{ padding: 20 }}>Loading post…</div>
        ) : error ? (
          <div className="card" role="alert" style={{ padding: 20, color: 'var(--danger, #d33)' }}>{error}</div>
        ) : post ? (
          <article className="card" style={{ padding: 22 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <strong>{post.username}</strong>
              <span style={{ color: 'var(--text-tertiary)' }}>{post.handle}</span>
              <span style={{ marginLeft: 'auto' }}>
                <RiskBadge level={post.riskLevel} score={post.riskScore} />
              </span>
            </div>
            <p style={{ margin: '18px 0 12px', lineHeight: 1.65, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
              {post.content}
            </p>
            <time style={{ color: 'var(--text-tertiary)', fontSize: '0.8rem' }}>
              {new Date(post.createdAt).toLocaleString()}
            </time>
          </article>
        ) : null}
      </div>
    </main>
  );
}
