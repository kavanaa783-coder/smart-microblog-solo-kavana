import { useCallback, useEffect, useState } from 'react';
import { Routes, Route, Outlet } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import Explore from './pages/Explore';
import Notifications from './pages/Notifications';
import Messages from './pages/Messages';
import Bookmarks from './pages/Bookmarks';
import Profile from './pages/Profile';
import Settings from './pages/Settings';
import PostDetail from './pages/PostDetail';

// ToastContext-lite: App owns the single toast so every page/route can share it
// without prop-drilling through the router. Pages call it via the Outlet context.
function Layout() {
  const [toastMsg, setToastMsg] = useState('');
  const [profileRevision, setProfileRevision] = useState(0);

  useEffect(() => {
    const refreshProfile = () => setProfileRevision((revision) => revision + 1);
    window.addEventListener('profile-updated', refreshProfile);
    return () => window.removeEventListener('profile-updated', refreshProfile);
  }, []);

  const showToast = useCallback((msg) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(''), 2400);
  }, []);

  return (
    <div className="app-shell">
      <Sidebar onPlaceholderClick={(label) => showToast(`${label} — check it out`)} />
      <Outlet key={profileRevision} context={{ showToast }} />
      {toastMsg && <div className="toast">{toastMsg}</div>}
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="explore" element={<Explore />} />
        <Route path="notifications" element={<Notifications />} />
        <Route path="messages" element={<Messages />} />
        <Route path="bookmarks" element={<Bookmarks />} />
        <Route path="profile" element={<Profile />} />
        <Route path="settings" element={<Settings />} />
        <Route path="post/:postId" element={<PostDetail />} />
      </Route>
    </Routes>
  );
}
