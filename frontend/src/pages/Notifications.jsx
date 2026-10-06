import { useOutletContext } from 'react-router-dom';
import { FiBell, FiHeart, FiMessageCircle, FiUserPlus } from 'react-icons/fi';

const items = [
  { type: 'like', user: 'sneha', text: 'liked your post about privacy-first AI', icon: FiHeart, tone: 'var(--risk-low)' },
  { type: 'reply', user: 'arjun', text: 'replied to your thread: “Great point. I would also mask DOBs.”', icon: FiMessageCircle, tone: 'var(--brand)' },
  { type: 'follow', user: 'mahima', text: 'followed you and started a private discussion', icon: FiUserPlus, tone: 'var(--risk-medium)' },
];

export default function Notifications() {
  useOutletContext();

  return (
    <main className="main-col">
      <div className="page-header">
        <h1>Notifications</h1>
        <p>Updates from your network</p>
      </div>

      <div className="page-body" style={{ padding: 16 }}>
        <div style={{ display: 'grid', gap: 12 }}>
          {items.map(({ user, text, icon: Icon, tone }) => (
            <div key={`${user}-${text}`} className="card" style={{ padding: 18, display: 'flex', alignItems: 'flex-start', gap: 12 }}>
              <div style={{ width: 36, height: 36, borderRadius: 999, display: 'grid', placeItems: 'center', background: 'var(--brand-soft)', color: tone }}>
                <Icon size={16} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700 }}>{user}</div>
                <div style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>{text}</div>
              </div>
              <FiBell size={16} color="var(--text-tertiary)" />
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
