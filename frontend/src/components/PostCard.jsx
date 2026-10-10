import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useOutletContext } from 'react-router-dom';
import {
  FiHeart,
  FiMessageCircle,
  FiRepeat,
  FiShare,
  FiBookmark,
  FiTrash2,
} from 'react-icons/fi';
import RiskBadge from './RiskBadge';
import {
  createReply,
  getReplies,
  toggleBookmark,
  removeBookmark,
  toggleLike,
  toggleRepost,
} from '../api/client';
import './PostCard.css';

function timeAgo(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);

  if (mins < 1) return 'now';
  if (mins < 60) return `${mins}m`;

  const hrs = Math.floor(mins / 60);

  if (hrs < 24) return `${hrs}h`;

  return `${Math.floor(hrs / 24)}d`;
}

export default function PostCard({ post, onDelete }) {
  const outletContext = useOutletContext();
  const showToast = outletContext?.showToast;
  const [liked, setLiked] = useState(Boolean(post.likedByUser));
  const [likeCount, setLikeCount] = useState(post.likeCount || 0);
  const [reposted, setReposted] = useState(Boolean(post.repostedByUser));
  const [repostCount, setRepostCount] = useState(post.repostCount || 0);
  const [updatingLike, setUpdatingLike] = useState(false);
  const [updatingRepost, setUpdatingRepost] = useState(false);
  const [actionError, setActionError] = useState('');
  const [bookmarked, setBookmarked] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const [showReplies, setShowReplies] = useState(false);
  const [replies, setReplies] = useState([]);
  const [replyText, setReplyText] = useState('');
  const [loadingReplies, setLoadingReplies] = useState(false);
  const [sendingReply, setSendingReply] = useState(false);
  const [replyError, setReplyError] = useState('');

  const avatarUrl = `https://api.dicebear.com/7.x/thumbs/svg?seed=${encodeURIComponent(
    post.handle || post.username || String(post.id)
  )}`;

  useEffect(() => {
    if (!showReplies) return;

    const loadReplies = async () => {
      setLoadingReplies(true);
      setReplyError('');

      try {
        const data = await getReplies(post.id);
        setReplies(data);
      } catch {
        setReplyError('Could not load replies.');
      } finally {
        setLoadingReplies(false);
      }
    };

    loadReplies();
  }, [showReplies, post.id]);

  const handleDeleteClick = () => {
    if (confirmingDelete) {
      onDelete(post.id);
    } else {
      setConfirmingDelete(true);
      setTimeout(() => setConfirmingDelete(false), 2500);
    }
  };

  const handleReplySubmit = async (event) => {
    event.preventDefault();

    const content = replyText.trim();

    if (!content || sendingReply) return;

    setSendingReply(true);
    setReplyError('');

    try {
      const newReply = await createReply(post.id, content);

      setReplies((currentReplies) => [...currentReplies, newReply]);
      setReplyText('');
    } catch {
      setReplyError('Could not save reply.');
    } finally {
      setSendingReply(false);
    }
  };

  const handleBookmarkToggle = async () => {
    const nextValue = !bookmarked;
    setBookmarked(nextValue);

    try {
      if (nextValue) {
        await toggleBookmark({ postId: post.id });
      } else {
        await removeBookmark({ postId: post.id });
      }
    } catch {
      setBookmarked(!nextValue);
    }
  };

  const handleLikeToggle = async () => {
    if (updatingLike) return;
    const previousLiked = liked;
    const previousCount = likeCount;
    setLiked(!previousLiked);
    setLikeCount(Math.max(0, previousCount + (previousLiked ? -1 : 1)));
    setUpdatingLike(true);
    setActionError('');

    try {
      const result = await toggleLike(post.id);
      setLiked(result.liked);
      setLikeCount(result.like_count);
    } catch {
      setLiked(previousLiked);
      setLikeCount(previousCount);
      setActionError('Like could not be saved. Try again.');
    } finally {
      setUpdatingLike(false);
    }
  };

  const handleRepostToggle = async () => {
    if (updatingRepost) return;
    const previousReposted = reposted;
    const previousCount = repostCount;
    setReposted(!previousReposted);
    setRepostCount(Math.max(0, previousCount + (previousReposted ? -1 : 1)));
    setUpdatingRepost(true);
    setActionError('');

    try {
      const result = await toggleRepost(post.id);
      setReposted(result.reposted);
      setRepostCount(result.repost_count);
    } catch {
      setReposted(previousReposted);
      setRepostCount(previousCount);
      setActionError('Repost could not be saved. Try again.');
    } finally {
      setUpdatingRepost(false);
    }
  };

  const handleShare = async () => {
    const shareUrl = `${window.location.origin}/post/${post.id}`;

    try {
      if (navigator.share) {
        await navigator.share({
          title: 'Smart Microblog',
          text: post.content,
          url: shareUrl,
        });
        showToast?.('Post shared');
      } else {
        await copyText(shareUrl);
        showToast?.('Post link copied');
      }
    } catch (error) {
      if (error?.name === 'AbortError') return;
      setActionError('Could not share this post. Please try again.');
    }
  };

  return (
    <motion.article
      className="post-card glass-card glass-card--hover"
      layout
      initial={{ opacity: 0, y: -10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="post-card__row">
        <div className="post-card__avatar-ring">
          <img className="post-card__avatar" src={avatarUrl} alt="" />
        </div>

        <div className="post-card__main">
          <div className="post-card__meta">
            <span className="post-card__name">{post.username}</span>
            <span className="post-card__handle">
              {post.handle || `@${post.username}`}
            </span>
            <span className="post-card__dot">·</span>
            <span className="post-card__time">
              {timeAgo(post.createdAt)}
            </span>
            <span className="post-card__badge">
              <RiskBadge level={post.riskLevel} score={post.riskScore} />
            </span>
          </div>

          <p className="post-card__content">{post.content}</p>

          <div className="post-card__actions">
            <button
              className="post-card__action"
              onClick={() => setShowReplies((current) => !current)}
              title="Reply"
            >
              <FiMessageCircle size={16} />
              <span>Reply</span>
            </button>

            <button
              className={`post-card__action ${reposted ? 'post-card__action--saved' : ''}`}
              title={reposted ? 'Undo repost' : 'Repost'}
              onClick={handleRepostToggle}
              disabled={updatingRepost}
            >
              <FiRepeat size={16} />
              <span>{reposted ? 'Reposted' : 'Repost'}{repostCount > 0 ? ` · ${repostCount}` : ''}</span>
            </button>

            <button
              className={`post-card__action ${
                liked ? 'post-card__action--liked' : ''
              }`}
              onClick={handleLikeToggle}
              disabled={updatingLike}
              title="Like"
            >
              <FiHeart
                size={16}
                fill={liked ? 'currentColor' : 'none'}
              />
              <span>{likeCount > 0 ? likeCount : ''}</span>
            </button>

            <button
              className="post-card__action"
              onClick={handleShare}
              title="Share"
            >
              <FiShare size={16} />
            </button>

            <button
              className={`post-card__action ${
                bookmarked ? 'post-card__action--saved' : ''
              }`}
              onClick={handleBookmarkToggle}
              title="Bookmark"
            >
              <FiBookmark
                size={16}
                fill={bookmarked ? 'currentColor' : 'none'}
              />
            </button>

            <button
              className={`post-card__action post-card__action--danger ${
                confirmingDelete ? 'post-card__action--confirm' : ''
              }`}
              onClick={handleDeleteClick}
              title="Delete"
            >
              <FiTrash2 size={16} />
              {confirmingDelete && <span>Confirm</span>}
            </button>
          </div>

          {actionError && (
            <div role="status" className="post-card__reply-error">
              {actionError}
            </div>
          )}

          {showReplies && (
            <div className="post-card__replies">
              <form
                className="post-card__reply-form"
                onSubmit={handleReplySubmit}
              >
                <input
                  type="text"
                  value={replyText}
                  onChange={(event) => setReplyText(event.target.value)}
                  placeholder={`Reply to ${post.username}...`}
                  maxLength={500}
                  disabled={sendingReply}
                />

                <button
                  type="submit"
                  disabled={!replyText.trim() || sendingReply}
                >
                  {sendingReply ? 'Sending…' : 'Reply'}
                </button>
              </form>

              {replyError && (
                <div className="post-card__reply-error">
                  {replyError}
                </div>
              )}

              {loadingReplies ? (
                <div className="post-card__reply-status">
                  Loading replies…
                </div>
              ) : replies.length === 0 ? (
                <div className="post-card__reply-status">
                  No replies yet.
                </div>
              ) : (
                <div className="post-card__reply-list">
                  {replies.map((reply) => (
                    <div
                      key={reply.id}
                      className="post-card__reply"
                    >
                      <strong>@{reply.username}</strong>
                      <span>{reply.content}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.article>
  );
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(value);
      return;
    } catch {
      // Continue to the compatibility fallback below.
    }
  }

  const input = document.createElement('textarea');
  input.value = value;
  input.setAttribute('readonly', '');
  input.style.position = 'fixed';
  input.style.opacity = '0';
  document.body.appendChild(input);
  input.select();
  const copied = document.execCommand('copy');
  input.remove();

  if (!copied) {
    throw new Error('Clipboard copy was not available');
  }
}