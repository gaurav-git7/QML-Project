import { Icon } from './Icons.jsx'

export const NAV = [
  { id: 'read', label: 'Reading Room', short: 'Read', icon: 'read', title: 'Reading Room' },
  { id: 'worklist', label: 'Worklist & Validation', short: 'Worklist', icon: 'worklist', title: 'Worklist & Validation' },
  { id: 'perf', label: 'Model Performance', short: 'Models', icon: 'perf', title: 'Model Performance' },
  { id: 'heart', label: 'Cardiac Risk', short: 'Cardiac', icon: 'heart', title: 'Cardiac Risk' },
  { id: 'methods', label: 'Methods & Limits', short: 'Info', icon: 'methods', title: 'Methods & Limits' },
]

export default function NavRail({ page, setPage, ready, theme, toggleTheme }) {
  return (
    <nav className="rail" aria-label="Main navigation">
      <div className="rail-brand">
        <div className="mark" aria-hidden="true">Q</div>
        <div className="brand-text"><b>QuantumML</b><small>Diagnostic Workstation</small></div>
      </div>
      <ul>
        {NAV.map((n) => (
          <li key={n.id}>
            <button className="rail-item" aria-current={page === n.id ? 'page' : undefined} title={n.label} aria-label={n.label} onClick={() => setPage(n.id)}>
              <Icon name={n.icon} size={20} />
              <span className="full">{n.label}</span><span className="short">{n.short}</span>
            </button>
          </li>
        ))}
      </ul>
      <div className="rail-foot">
        <div className={'status' + (ready ? ' ok' : '')} role="status"><i /><span>{ready ? 'Models ready' : 'Loading models…'}</span></div>
        <button className="theme-btn" onClick={toggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} title="Switch theme">
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={18} /><span>{theme === 'dark' ? 'Light theme' : 'Dark theme'}</span>
        </button>
      </div>
    </nav>
  )
}
