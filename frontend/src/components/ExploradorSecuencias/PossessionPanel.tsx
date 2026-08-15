import type { SequenceEntry } from "../../types";

export default function PossessionPanel({ entry }: { entry: SequenceEntry }) {
  const pct = entry.possessionPct;
  if (!pct) {
    return (
      <div className="panel">
        <h2>Posesión de balón</h2>
        <div className="empty-note">No disponible para esta combinación.</div>
      </div>
    );
  }

  const p0 = pct["0"] ?? 0;
  const p1 = pct["1"] ?? 0;

  return (
    <div className="panel">
      <h2>Posesión de balón</h2>
      <p className="muted">
        La posesión se estima por proximidad: en cada fotograma con balón detectado,
        se asigna la posesión al equipo del jugador más cercano dentro de
        un radio de 150&nbsp;px.
      </p>
      <div className="possession-bar">
        <div className="team0" style={{ width: `${p0}%` }}>
          <span>{p0 > 8 ? `${p0.toFixed(0)}%` : ""}</span>
        </div>
        <div className="team1" style={{ width: `${p1}%` }}>
          <span>{p1 > 8 ? `${p1.toFixed(0)}%` : ""}</span>
        </div>
      </div>
      <div className="legend">
        <span>
          <i className="dot team0" /> Equipo 0 — {p0.toFixed(1)}%
        </span>
        <span>
          <i className="dot team1" /> Equipo 1 — {p1.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
