import type { SequenceEntry } from "../../types";

export default function TeamsPanel({ entry }: { entry: SequenceEntry }) {
  return (
    <div className="panel">
      <h2>Clasificación de equipos</h2>
      <p className="muted">
        Clasificación por K-means (k=2) sobre el color medio de la
        camiseta, muestreado cada 5 fotogramas.
      </p>
      <div className="stat-row">
        <div className="stat">
          <div className="value">{entry.playersClassified}</div>
          <div className="label">jugadores clasificados</div>
        </div>
        <div className="stat">
          <div className="value">{entry.teamCounts["0"]}</div>
          <div className="label">equipo 0</div>
        </div>
        <div className="stat">
          <div className="value">{entry.teamCounts["1"]}</div>
          <div className="label">equipo 1</div>
        </div>
      </div>
      <p className="footnote">
        El árbitro puede quedar asignado
        al equipo cuyo centroide de color le resulte más cercano: es una
        limitación conocida de la clasificación en k=2 (ver capítulo de
        Fundamentos teóricos de la memoria).
      </p>
    </div>
  );
}
