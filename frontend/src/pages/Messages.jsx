import { useCallback, useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { FiMessageCircle, FiSend } from 'react-icons/fi';
import { getConversations, getCurrentUsername, getMessages, sendMessage } from '../api/client';

function timeLabel(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

export default function Messages() {
  const { showToast } = useOutletContext();
  const username = getCurrentUsername();
  const [conversations, setConversations] = useState([]);
  const [activeUser, setActiveUser] = useState('');
  const [messages, setMessages] = useState([]);
  const [recipientInput, setRecipientInput] = useState('');
  const [draft, setDraft] = useState('');
  const [loadingConversations, setLoadingConversations] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');

  const refreshConversations = useCallback(async (preferredUser = '') => {
    try {
      const data = await getConversations(username);
      setConversations(data);
      const nextUser = preferredUser || activeUser || data[0]?.username || '';
      if (nextUser) setActiveUser(nextUser);
      setError('');
    } catch {
      setError('Could not load conversations. Check that the API and database are running.');
    } finally {
      setLoadingConversations(false);
    }
  }, [activeUser, username]);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  useEffect(() => {
    let cancelled = false;
    if (!activeUser) {
      setMessages([]);
      return undefined;
    }

    setLoadingMessages(true);
    getMessages(activeUser, username)
      .then((data) => {
        if (!cancelled) {
          setMessages(data);
          setError('');
        }
      })
      .catch(() => {
        if (!cancelled) setError('Could not load this conversation.');
      })
      .finally(() => {
        if (!cancelled) setLoadingMessages(false);
      });

    return () => { cancelled = true; };
  }, [activeUser, username]);

  const startConversation = (event) => {
    event.preventDefault();
    const target = recipientInput.trim().replace(/^@/, '');
    if (!target) return;
    if (target.toLowerCase() === username.toLowerCase()) {
      showToast('Choose another username to message');
      return;
    }
    setActiveUser(target);
    setRecipientInput('');
    setMessages([]);
  };

  const handleSend = async (event) => {
    event.preventDefault();
    const content = draft.trim();
    if (!activeUser || !content || sending) return;

    setSending(true);
    setError('');
    try {
      const saved = await sendMessage(activeUser, content, username);
      setMessages((current) => [...current, saved]);
      setDraft('');
      await refreshConversations(activeUser);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(detail || 'Message could not be sent. Check the API and database, then try again.');
    } finally {
      setSending(false);
    }
  };

  return (
    <main className="main-col">
      <div className="page-header">
        <h1>Messages</h1>
        <p>Enter a username to start a private conversation and send a message.</p>
      </div>

      <div className="page-body" style={{ display: 'grid', gridTemplateColumns: 'minmax(220px, 0.75fr) minmax(0, 1.6fr)', minHeight: 560 }}>
        <aside style={{ borderRight: '1px solid var(--border)', padding: 16 }}>
          <form onSubmit={startConversation} style={{ display: 'grid', gap: 8, marginBottom: 18 }}>
            <label htmlFor="message-recipient" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Start a conversation</label>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                id="message-recipient"
                value={recipientInput}
                onChange={(event) => setRecipientInput(event.target.value)}
                placeholder="Recipient username"
                autoComplete="off"
                style={{ width: 0, minWidth: 0, flex: '1 1 0%', border: '1px solid var(--border)', background: 'var(--surface-alt)', color: 'var(--text-primary)', borderRadius: 10, padding: '10px 12px' }}
              />
              <button type="submit" aria-label="Open conversation" title="Open conversation" style={{ border: 0, borderRadius: 10, padding: '0 12px', background: 'var(--brand)', color: '#fff', cursor: 'pointer' }}>
                <FiMessageCircle />
              </button>
            </div>
          </form>

          <strong style={{ display: 'block', marginBottom: 10 }}>Conversations</strong>
          {loadingConversations ? <p style={{ color: 'var(--text-tertiary)' }}>Loading…</p> : null}
          {!loadingConversations && conversations.length === 0 ? (
            <p style={{ color: 'var(--text-tertiary)', fontSize: '0.85rem', lineHeight: 1.5 }}>No messages yet. Enter a username above to start one.</p>
          ) : null}
          <div style={{ display: 'grid', gap: 6 }}>
            {conversations.map((conversation) => (
              <button
                key={conversation.username}
                type="button"
                onClick={() => setActiveUser(conversation.username)}
                style={{ textAlign: 'left', border: 0, borderRadius: 12, padding: 12, cursor: 'pointer', color: 'var(--text-primary)', background: activeUser === conversation.username ? 'var(--brand-soft)' : 'transparent' }}
              >
                <strong>{conversation.username}</strong>
                <span style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: 4, color: 'var(--text-secondary)', fontSize: '0.78rem' }}>{conversation.last_message}</span>
                <span style={{ display: 'block', marginTop: 4, color: 'var(--text-tertiary)', fontSize: '0.7rem' }}>{timeLabel(conversation.updated_at)}</span>
              </button>
            ))}
          </div>
        </aside>

        <section style={{ padding: 16, display: 'flex', minWidth: 0, flexDirection: 'column', minHeight: 520 }}>
          {activeUser ? (
            <>
              <header style={{ padding: '8px 0 14px', borderBottom: '1px solid var(--border)' }}>
                <strong>{activeUser}</strong>
                <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', marginTop: 3 }}>Conversation with @{activeUser}</div>
              </header>

              <div aria-live="polite" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto', padding: '18px 2px' }}>
                {loadingMessages ? <p style={{ color: 'var(--text-tertiary)' }}>Loading messages…</p> : null}
                {!loadingMessages && messages.length === 0 ? (
                  <div style={{ margin: 'auto', maxWidth: 300, textAlign: 'center', color: 'var(--text-tertiary)' }}>
                    <FiMessageCircle size={26} />
                    <p>No messages in this conversation yet. Say hello below.</p>
                  </div>
                ) : null}
                {messages.map((message) => {
                  const mine = message.sender === username;
                  return (
                    <div key={message.id} style={{ display: 'flex', justifyContent: mine ? 'flex-end' : 'flex-start' }}>
                      <div style={{ maxWidth: '78%', borderRadius: 14, padding: '10px 13px', background: mine ? 'var(--brand)' : 'var(--surface-alt)', color: mine ? '#fff' : 'var(--text-primary)', overflowWrap: 'anywhere' }}>
                        <div>{message.content}</div>
                        <time style={{ display: 'block', marginTop: 5, fontSize: '0.68rem', opacity: 0.75 }}>{timeLabel(message.created_at)}</time>
                      </div>
                    </div>
                  );
                })}
              </div>

              <form onSubmit={handleSend} style={{ display: 'flex', gap: 10, borderTop: '1px solid var(--border)', paddingTop: 14 }}>
                <input
                  aria-label="Message text"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder={`Message @${activeUser}`}
                  maxLength={2000}
                  style={{ width: 0, minWidth: 0, flex: '1 1 0%', border: '1px solid var(--border)', background: 'var(--surface-alt)', color: 'var(--text-primary)', borderRadius: 999, padding: '12px 16px' }}
                />
                <button className="composer__post-btn" type="submit" disabled={sending || !draft.trim()} aria-label="Send message">
                  <FiSend style={{ marginRight: 6, verticalAlign: 'middle' }} />{sending ? 'Sending…' : 'Send'}
                </button>
              </form>
            </>
          ) : (
            <div style={{ margin: 'auto', textAlign: 'center', maxWidth: 340, color: 'var(--text-tertiary)' }}>
              <FiMessageCircle size={32} />
              <h2 style={{ color: 'var(--text-primary)', fontSize: '1.1rem' }}>Your messages</h2>
              <p>Choose an existing conversation or enter a username to start a new one.</p>
            </div>
          )}
          {error ? <p role="alert" style={{ color: 'var(--danger, #d33)', marginBottom: 0 }}>{error}</p> : null}
        </section>
      </div>
    </main>
  );
}
