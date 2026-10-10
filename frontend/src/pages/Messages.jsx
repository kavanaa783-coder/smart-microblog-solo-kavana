import { useCallback, useEffect, useRef, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { FiMessageCircle, FiSend } from 'react-icons/fi';
import { getConversations, getCurrentUsername, getMessages, sendMessage } from '../api/client';
import './Messages.css';

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
  const messageListRef = useRef(null);

  useEffect(() => {
    messageListRef.current?.scrollTo({
      top: messageListRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages]);

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
      showToast('Message sent');
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
        <p>Your private conversations, saved between visits.</p>
      </div>

      <div className="page-body messages-page">
        <aside className="messages-sidebar">
          <form onSubmit={startConversation} className="messages-start-form">
            <label htmlFor="message-recipient" className="messages-section-label">New conversation</label>
            <div className="messages-recipient-control">
              <input
                id="message-recipient"
                value={recipientInput}
                onChange={(event) => setRecipientInput(event.target.value)}
                placeholder="Username"
                autoComplete="off"
                aria-label="Recipient username"
              />
              <button type="submit" className="messages-icon-button" aria-label="Open conversation" title="Open conversation" disabled={!recipientInput.trim()}>
                <FiMessageCircle size={15} />
                <span>Open</span>
              </button>
            </div>
          </form>

          <div className="messages-list-heading">
            <h2>Conversations</h2>
            {!loadingConversations && <span>{conversations.length}</span>}
          </div>
          {loadingConversations ? <p className="messages-muted">Loading conversations…</p> : null}
          {!loadingConversations && conversations.length === 0 ? (
            <p className="messages-muted">No conversations yet. Start one above.</p>
          ) : null}
          <div className="messages-conversation-list">
            {conversations.map((conversation) => (
              <button
                key={conversation.username}
                type="button"
                onClick={() => setActiveUser(conversation.username)}
                className={`messages-conversation ${activeUser === conversation.username ? 'messages-conversation--active' : ''}`}
              >
                <span className="messages-avatar" aria-hidden="true">{conversation.username.slice(0, 1).toUpperCase()}</span>
                <span className="messages-conversation-copy">
                  <strong>{conversation.username}</strong>
                  <span className="messages-preview">{conversation.last_message}</span>
                </span>
                <time>{timeLabel(conversation.updated_at)}</time>
              </button>
            ))}
          </div>
        </aside>

        <section className="messages-thread">
          {activeUser ? (
            <>
              <header className="messages-thread-header">
                <span className="messages-avatar messages-avatar--large" aria-hidden="true">{activeUser.slice(0, 1).toUpperCase()}</span>
                <div>
                  <strong>{activeUser}</strong>
                  <span>Private conversation</span>
                </div>
              </header>

              <div className="messages-scroll" aria-live="polite" ref={messageListRef}>
                {loadingMessages ? <p className="messages-muted messages-status">Loading messages…</p> : null}
                {!loadingMessages && messages.length === 0 ? (
                  <div className="messages-empty-thread">
                    <FiMessageCircle size={26} />
                    <strong>No messages yet</strong>
                    <p>Send the first message to {activeUser}.</p>
                  </div>
                ) : null}
                {messages.map((message) => {
                  const mine = message.sender === username;
                  return (
                    <div key={message.id} className={`messages-row ${mine ? 'messages-row--mine' : ''}`}>
                      <div className={`messages-bubble ${mine ? 'messages-bubble--mine' : ''}`}>
                        <p>{message.content}</p>
                        <time>{mine ? 'You · ' : ''}{timeLabel(message.created_at)}</time>
                      </div>
                    </div>
                  );
                })}
              </div>

              <form onSubmit={handleSend} className="messages-compose">
                <input
                  aria-label="Message text"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder={`Message @${activeUser}`}
                  maxLength={2000}
                  disabled={sending}
                />
                <button className="composer__post-btn" type="submit" disabled={sending || !draft.trim()} aria-label="Send message">
                  <FiSend size={16} /> <span>{sending ? 'Sending…' : 'Send'}</span>
                </button>
              </form>
            </>
          ) : (
            <div className="messages-empty-thread">
              <FiMessageCircle size={32} />
              <strong>Your messages</strong>
              <p>Select a conversation or enter a username to start a new one.</p>
            </div>
          )}
          {error ? <p role="alert" className="messages-error">{error}</p> : null}
        </section>
      </div>
    </main>
  );
}
