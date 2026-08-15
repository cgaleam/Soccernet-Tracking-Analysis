import type { SequenceEntry } from "../../types";

const ALGO_LABEL: Record<string, string> = {
  bytetracker: "ByteTrack",
  ocsort: "OC-SORT",
};

export default function VideoPanel({ entry }: { entry: SequenceEntry }) {
  return (
    <div className="panel">
      <h2>Secuencia de vídeo</h2>
      <p className="muted">
        Tramo de frames: [{entry.videoFrameRange[0]}–{entry.videoFrameRange[1]}] de{" "}
        {entry.sequence}, tracking de {ALGO_LABEL[entry.algorithm]} con
        color por equipo (azul/rojo) y balón amarillo.
        Reconstruido a partir del resultado de tracking ya calculado, sin
        volver a ejecutar el tracker.
      </p>
      <video
        className="video-frame"
        key={entry.videoUrl}
        src={entry.videoUrl}
        controls
        muted
        loop
        playsInline
      />
    </div>
  );
}
