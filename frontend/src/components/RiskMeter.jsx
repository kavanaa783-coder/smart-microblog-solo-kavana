import { motion } from 'framer-motion';
import './RiskMeter.css';

const RISK_COLOR = {
  LOW: 'var(--risk-low)',
  MEDIUM: 'var(--risk-medium)',
  HIGH: 'var(--risk-high)',
};

const RADIUS = 24;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function RiskMeter({ scanning, result, idle }) {
  const score = result?.riskScore ?? 0;
  const level = result?.riskLevel ?? 'LOW';

  const pct = Math.min(100, Math.max(0, score));
  const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
  const color = RISK_COLOR[level] || RISK_COLOR.LOW;

  const entities = result?.entities || {};

  const detectedTypes = Object.entries(entities).filter(([, values]) =>
    Array.isArray(values) ? values.length > 0 : Boolean(values)
  );

  return (
    <div className="risk-meter">
      <div className="risk-meter__gauge">
        <svg viewBox="0 0 60 60">
          <circle
            className="risk-meter__track"
            cx="30"
            cy="30"
            r={RADIUS}
          />

          <motion.circle
            className="risk-meter__arc"
            cx="30"
            cy="30"
            r={RADIUS}
            stroke={color}
            strokeDasharray={CIRCUMFERENCE}
            animate={{
              strokeDashoffset: idle ? CIRCUMFERENCE : offset,
            }}
            transition={{
              duration: 0.5,
              ease: 'easeOut',
            }}
          />
        </svg>

        <span className="risk-meter__score">
          {idle ? '—' : score}
        </span>
      </div>

      <div className="risk-meter__body">
        <div className="risk-meter__label">
          {scanning && <span className="risk-meter__pulse" />}

          {scanning
            ? 'Scanning as you type…'
            : 'Live Risk Meter'}
        </div>

        <div
          className="risk-meter__level"
          style={{
            color: idle
              ? 'var(--text-tertiary)'
              : color,
          }}
        >
          {idle ? 'Start typing to see risk' : level}
        </div>

        {detectedTypes.length > 0 && (
          <div className="risk-meter__entities">
            {detectedTypes.map(([type, values]) => (
              <span
                key={type}
                className="risk-meter__chip"
              >
                ✔ {formatEntityLabel(type)}:{' '}
                {Array.isArray(values)
                  ? values.join(', ')
                  : values}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function formatEntityLabel(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}