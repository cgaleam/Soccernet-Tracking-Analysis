// Logotipo del proyecto: un campo de fútbol visto en planta, con el
// círculo central resaltado en color verde.

export default function PitchLogo({ size = 64 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={(size * 90) / 140}
      viewBox="0 0 140 90"
      role="img"
      aria-label="Logotipo: campo de fútbol"
    >
      <rect x="5" y="5" width="130" height="80" rx="2" fill="var(--bg-panel-alt)" stroke="var(--text-muted)" strokeWidth="1.5" />
      {/* línea de medio campo */}
      <line x1="70" y1="5" x2="70" y2="85" stroke="var(--text-muted)" strokeWidth="1.5" />
      {/* áreas */}
      <rect x="5" y="24" width="16" height="42" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" />
      <rect x="119" y="24" width="16" height="42" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" />
      <rect x="5" y="34" width="6" height="22" fill="none" stroke="var(--text-muted)" strokeWidth="1.2" />
      <rect x="129" y="34" width="6" height="22" fill="none" stroke="var(--text-muted)" strokeWidth="1.2" />
      {/* círculo central: color de acento */}
      <circle cx="70" cy="45" r="15" fill="none" stroke="var(--accent)" strokeWidth="2" />
      <circle cx="70" cy="45" r="2.4" fill="var(--accent)" />
    </svg>
  );
}
