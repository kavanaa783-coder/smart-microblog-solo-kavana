import { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { getCurrentUsername, getProfile, saveProfile } from '../api/client';

export default function Profile() {
  const { showToast } = useOutletContext();
  const [username, setUsername] = useState(getCurrentUsername());
  const [bio, setBio] = useState('');
  const [image, setImage] = useState('');
  const [savedProfile, setSavedProfile] = useState({ username: getCurrentUsername(), bio: '', image: '' });
  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    const currentUsername = getCurrentUsername();

    getProfile(currentUsername)
      .then((profile) => {
        if (!active) return;
        setUsername(profile.username || currentUsername);
        setBio(profile.bio || '');
        setImage(profile.profileImage || '');
        setSavedProfile({
          username: profile.username || currentUsername,
          bio: profile.bio || '',
          image: profile.profileImage || '',
        });
      })
      .catch(() => {})
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setSaving(true);

    try {
      const saved = await saveProfile({
        username: username.trim().replace(/^@/, ''),
        currentUsername: savedProfile.username,
        bio: bio.trim(),
        profileImage: image.trim(),
      });

      setUsername(saved.username || username);
      setBio(saved.bio || '');
      setImage(saved.profileImage || '');
      setSavedProfile({
        username: saved.username || username,
        bio: saved.bio || '',
        image: saved.profileImage || '',
      });
      setIsEditing(false);
      showToast('Profile saved');
    } catch (error) {
      showToast(error?.response?.data?.detail || 'Could not save profile');
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setUsername(savedProfile.username);
    setBio(savedProfile.bio);
    setImage(savedProfile.image);
    setIsEditing(false);
  };

  const avatar = image || `https://api.dicebear.com/7.x/thumbs/svg?seed=${encodeURIComponent(username)}`;

  return (
    <main className="main-col">
      <div className="page-header">
        <h1>Profile</h1>
        <p>Manage your public identity</p>
      </div>

      <div className="page-body" style={{ padding: 24 }}>
        {loading ? (
          <div style={{ padding: 24, color: 'var(--text-tertiary)' }}>Loading profile…</div>
        ) : !isEditing ? (
          <section className="card" style={{ padding: 24, maxWidth: 640 }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, minWidth: 0 }}>
                <img
                  src={avatar}
                  alt={`${username} profile`}
                  style={{ width: 76, height: 76, borderRadius: '50%', objectFit: 'cover', background: 'var(--brand-soft)' }}
                />
                <div style={{ minWidth: 0 }}>
                  <h2 style={{ fontSize: '1.2rem', overflowWrap: 'anywhere' }}>{username}</h2>
                  <div style={{ color: 'var(--text-tertiary)', fontSize: '0.88rem' }}>@{username}</div>
                </div>
              </div>
              <button type="button" className="composer__post-btn" onClick={() => setIsEditing(true)}>
                Edit profile
              </button>
            </div>
            <p style={{ marginTop: 20, lineHeight: 1.6, whiteSpace: 'pre-wrap', color: bio ? 'var(--text-primary)' : 'var(--text-tertiary)' }}>
              {bio || 'No bio yet.'}
            </p>
          </section>
        ) : (
          <form onSubmit={handleSubmit} style={{ display: 'grid', gap: 18, maxWidth: 640 }}>
            <div className="card" style={{ padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <img
                  src={avatar}
                  alt={`${username} profile`}
                  style={{ width: 64, height: 64, borderRadius: '50%', objectFit: 'cover', background: 'var(--brand-soft)' }}
                />
                <div>
                  <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{username}</div>
                  <div style={{ color: 'var(--text-tertiary)', fontSize: '0.8rem' }}>@{username}</div>
                </div>
              </div>
            </div>

            <div className="card" style={{ padding: 20, display: 'grid', gap: 14 }}>
              <label style={{ display: 'grid', gap: 8 }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Username</span>
                <input
                  value={username}
                  onChange={(event) => setUsername(event.target.value.replace(/^@/, ''))}
                  maxLength={30}
                  pattern="[A-Za-z0-9_]{1,30}"
                  title="Use 1–30 letters, numbers, or underscores."
                  required
                  autoComplete="username"
                  style={{ padding: '12px 14px', borderRadius: 12, border: '1px solid var(--border)', background: 'var(--surface-alt)', color: 'var(--text-primary)' }}
                />
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Your profile and existing posts will use this username.</span>
              </label>

              <label style={{ display: 'grid', gap: 8 }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Profile image URL</span>
                <input
                  value={image}
                  onChange={(event) => setImage(event.target.value)}
                  placeholder="https://..."
                  style={{ padding: '12px 14px', borderRadius: 12, border: '1px solid var(--border)', background: 'var(--surface-alt)', color: 'var(--text-primary)' }}
                />
              </label>

              <label style={{ display: 'grid', gap: 8 }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Bio</span>
                <textarea
                  value={bio}
                  onChange={(event) => setBio(event.target.value)}
                  rows={5}
                  maxLength={280}
                  style={{ padding: '12px 14px', borderRadius: 12, border: '1px solid var(--border)', background: 'var(--surface-alt)', resize: 'vertical', color: 'var(--text-primary)' }}
                />
                <span style={{ textAlign: 'right', fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>{bio.length}/280</span>
              </label>

              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <button type="submit" className="composer__post-btn" disabled={saving}>
                  {saving ? 'Saving…' : 'Save profile'}
                </button>
                <button type="button" onClick={handleCancel} disabled={saving} style={{ padding: '10px 16px', borderRadius: 999, border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                  Cancel
                </button>
              </div>
            </div>
          </form>
        )}
      </div>
    </main>
  );
}
