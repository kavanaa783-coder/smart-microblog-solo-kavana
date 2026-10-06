import { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';

const STORAGE_KEY = 'privacy-settings';
const defaultSettings = [
  { name: 'Auto-scan every post', value: true },
  { name: 'Warn before publishing risky posts', value: true },
  { name: 'Email alerts for activity', value: false },
  { name: 'Hide exact PII in analytics', value: true },
];

export default function Settings() {
  const { showToast } = useOutletContext();
  const [settings, setSettings] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
      if (!Array.isArray(saved) || saved.length === 0) {
        return defaultSettings;
      }
      return saved.map((item) => ({
        name: item.name,
        value: Boolean(item.value),
      }));
    } catch {
      return defaultSettings;
    }
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const toggleSetting = (name) => {
    setSettings((current) =>
      current.map((item) =>
        item.name === name ? { ...item, value: !item.value } : item
      )
    );
    showToast('Settings updated');
  };

  return (
    <main className="main-col">
      <div className="page-header">
        <h1>Settings</h1>
        <p>Privacy controls and preferences</p>
      </div>

      <div className="page-body" style={{ padding: 20 }}>
        <div className="card" style={{ padding: 20, display: 'grid', gap: 14 }}>
          {settings.map((setting) => (
            <div key={setting.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
              <span>{setting.name}</span>
              <button
                type="button"
                aria-pressed={setting.value}
                onClick={() => toggleSetting(setting.name)}
                style={{
                  width: 52,
                  height: 30,
                  borderRadius: 999,
                  border: 'none',
                  background: setting.value ? 'var(--brand)' : 'var(--surface-alt)',
                  position: 'relative',
                  padding: 0,
                  cursor: 'pointer',
                }}
              >
                <span
                  style={{
                    position: 'absolute',
                    top: 4,
                    left: setting.value ? 28 : 4,
                    width: 20,
                    height: 20,
                    borderRadius: '50%',
                    background: '#fff',
                    transition: 'left 0.2s ease',
                  }}
                />
              </button>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
