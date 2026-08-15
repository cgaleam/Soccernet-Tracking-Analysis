import type { SequenceEntry } from "../../types";

export default function HomografiaPanel({ entry }: { entry: SequenceEntry }) {
  if (!entry.heatmapHomography && !entry.topDistanceSpeed) {
    return (
      <div className="panel">
        <h2>Homografía: distancia y velocidad</h2>
        <div className="empty-note">
          {entry.sequence} no tiene puntos de calibración de homografía
          (solo se calibraron manualmente SNMOT-116 y SNMOT-118, ver
          capítulo de Implementación).
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Homografía: distancia y velocidad</h2>
      <p className="muted">
        Proyección de las posiciones de cada jugador al plano real del
        campo (metros) mediante una homografía calibrada manualmente con
        4 puntos de referencia sobre este tramo de vídeo.
      </p>
      <div className="panel-grid">
        {entry.heatmapHomography && (
          <div>
            <h3>Mapa de calor calibrado</h3>
            <img
              className="img-frame"
              src={entry.heatmapHomography}
              alt="Mapa de calor con homografía real"
            />
          </div>
        )}
        {entry.topDistanceSpeed && (
          <div>
            <h3>Top 5 distancia recorrida</h3>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Distancia (m)</th>
                  <th>Vel. media (km/h)</th>
                  <th>Vel. punta (km/h)</th>
                </tr>
              </thead>
              <tbody>
                {entry.topDistanceSpeed.map((row) => (
                  <tr key={row.track_id}>
                    <td>{row.track_id}</td>
                    <td>{row.distance_m.toFixed(1)}</td>
                    <td>{row.avg_speed_kmh.toFixed(1)}</td>
                    <td>{row.peak_speed_kmh.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
