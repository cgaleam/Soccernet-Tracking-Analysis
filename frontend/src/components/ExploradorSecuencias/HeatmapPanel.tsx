import type { SequenceEntry } from "../../types";

export default function HeatmapPanel({ entry }: { entry: SequenceEntry }) {
  return (
    <div className="panel">
      <h2>Mapas de calor de posiciones</h2>
      <p className="muted">
        Densidad de posiciones (KDE) proyectada sobre el campo,
        escalando por el rango de píxeles observado (aproximación sin
        homografía real; ver panel de homografía para la proyección
        calibrada donde existe).
      </p>
      <div className="panel-grid">
        <div>
          <h3>General</h3>
          {entry.heatmapGeneral ? (
            <img className="img-frame" src={entry.heatmapGeneral} alt="Mapa de calor general" />
          ) : (
            <div className="empty-note">No disponible para esta combinación.</div>
          )}
        </div>
        <div>
          <h3>Por equipo</h3>
          {entry.heatmapByTeam ? (
            <img className="img-frame" src={entry.heatmapByTeam} alt="Mapa de calor por equipo" />
          ) : (
            <div className="empty-note">
              No disponible: esta combinación de secuencia y algoritmo no
              tiene mapa de calor por equipo generado en el pipeline
              original.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
