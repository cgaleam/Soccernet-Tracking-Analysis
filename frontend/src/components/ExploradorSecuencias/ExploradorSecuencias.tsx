import { useEffect, useMemo, useState } from "react";
import { loadSequences } from "../../data/loadData";
import type { Algorithm, SequenceEntry } from "../../types";
import VideoPanel from "./VideoPanel";
import TeamsPanel from "./TeamsPanel";
import HeatmapPanel from "./HeatmapPanel";
import HomografiaPanel from "./HomografiaPanel";
import PossessionPanel from "./PossessionPanel";

const SEQUENCES = ["SNMOT-116", "SNMOT-117", "SNMOT-118"];
const ALGOS: { value: Algorithm; label: string }[] = [
  { value: "bytetracker", label: "ByteTracker" },
  { value: "ocsort", label: "OC-SORT" },
];

export default function ExploradorSecuencias() {
  const [entries, setEntries] = useState<SequenceEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sequence, setSequence] = useState(SEQUENCES[0]);
  const [algorithm, setAlgorithm] = useState<Algorithm>("bytetracker");

  useEffect(() => {
    loadSequences().then(setEntries).catch((e) => setError(String(e)));
  }, []);

  const entry = useMemo(
    () => entries?.find((e) => e.sequence === sequence && e.algorithm === algorithm) ?? null,
    [entries, sequence, algorithm]
  );

  if (error) return <div className="error">{error}</div>;
  if (!entries) return <div className="loading">Cargando secuencias…</div>;

  return (
    <>
      <div className="selectors">
        <div className="selector-group">
          <label>Secuencia</label>
          <div className="chip-group">
            {SEQUENCES.map((seq) => (
              <button
                key={seq}
                className={`chip ${sequence === seq ? "active" : ""}`}
                onClick={() => setSequence(seq)}
              >
                {seq}
              </button>
            ))}
          </div>
        </div>
        <div className="selector-group">
          <label>Algoritmo</label>
          <div className="chip-group">
            {ALGOS.map((a) => (
              <button
                key={a.value}
                className={`chip ${algorithm === a.value ? "active" : ""}`}
                onClick={() => setAlgorithm(a.value)}
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {!entry ? (
        <div className="empty-note">
          No hay datos para {sequence} / {algorithm}.
        </div>
      ) : (
        <>
          <VideoPanel entry={entry} />
          <div className="panel-grid">
            <TeamsPanel entry={entry} />
            <PossessionPanel entry={entry} />
          </div>
          <HeatmapPanel entry={entry} />
          <HomografiaPanel entry={entry} />
        </>
      )}
    </>
  );
}
