import { useEffect, useState } from "react";
import { loadGlobalMetrics } from "../../data/loadData";
import type { GlobalMetrics } from "../../types";

const ALGO_LABEL: Record<string, string> = {
  bytetracker: "ByteTrack",
  ocsort: "OC-SORT",
};

function MetricsTable({
  title,
  metrics,
}: {
  title: string;
  metrics: GlobalMetrics["bySplit"]["train"];
}) {
  const best = (key: "HOTA" | "HOTA0" | "MOTA" | "IDF1") =>
    metrics.bytetracker[key] >= metrics.ocsort[key] ? "bytetracker" : "ocsort";

  const cell = (algo: "bytetracker" | "ocsort", key: "HOTA" | "HOTA0" | "MOTA" | "IDF1" | "DetA" | "AssA" | "IDSW") => {
    const value = metrics[algo][key];
    const isBest = (key === "IDSW"
      ? (metrics.bytetracker.IDSW <= metrics.ocsort.IDSW ? "bytetracker" : "ocsort")
      : best(key as "HOTA" | "HOTA0" | "MOTA" | "IDF1")) === algo && key !== "DetA" && key !== "AssA";
    return <td className={isBest ? "best" : ""}>{value.toFixed(key === "IDSW" ? 0 : 2)}</td>;
  };

  return (
    <div>
      <h3>{title}</h3>
      <table>
        <thead>
          <tr>
            <th>Algoritmo</th>
            <th>HOTA</th>
            <th>HOTA(0)</th>
            <th>DetA</th>
            <th>AssA</th>
            <th>MOTA</th>
            <th>IDF1</th>
            <th>IDSW</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>ByteTrack</td>
            {cell("bytetracker", "HOTA")}
            {cell("bytetracker", "HOTA0")}
            {cell("bytetracker", "DetA")}
            {cell("bytetracker", "AssA")}
            {cell("bytetracker", "MOTA")}
            {cell("bytetracker", "IDF1")}
            {cell("bytetracker", "IDSW")}
          </tr>
          <tr>
            <td>OC-SORT</td>
            {cell("ocsort", "HOTA")}
            {cell("ocsort", "HOTA0")}
            {cell("ocsort", "DetA")}
            {cell("ocsort", "AssA")}
            {cell("ocsort", "MOTA")}
            {cell("ocsort", "IDF1")}
            {cell("ocsort", "IDSW")}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default function ComparativaAlgoritmos() {
  const [data, setData] = useState<GlobalMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadGlobalMetrics().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div className="loading">Cargando métricas…</div>;

  return (
    <>
      <div className="panel">
        <h2>HOTA frente a HOTA(0)</h2>
        <p className="muted">
          HOTA(0) es HOTA evaluado en un único umbral de IoU (0.5); HOTA
          integra el promedio en el rango 0.05–0.95. La configuración
          óptima de hiperparámetros de este proyecto se seleccionó con
          HOTA(0), lo que invierte el orden entre algoritmos respecto a
          HOTA integrado: bajo HOTA(0) ByteTrack aventaja a OC-SORT en
          test, pero bajo HOTA integrado es OC-SORT quien gana en ambos
          splits, porque OC-SORT reutiliza las detecciones sin
          interpolarlas (LocA/DetPr casi perfectos) mientras que
          ByteTrack sí interpola en su segunda ronda de asociación, lo
          que le penaliza en los umbrales de IoU más estrictos.
        </p>
        <div className="panel-grid">
          <MetricsTable title="Split train" metrics={data.bySplit.train} />
          <MetricsTable title="Split test" metrics={data.bySplit.test} />
        </div>
        {data.hotaVsHota0Chart && (
          <img
            className="img-frame"
            style={{ marginTop: "1rem" }}
            src={data.hotaVsHota0Chart}
            alt="Comparativa HOTA vs HOTA(0) por algoritmo y split"
          />
        )}
      </div>

      <div className="panel">
        <h2>Configuración óptima de hiperparámetros</h2>
        <p className="muted">
          Aquí se muestra la mejor combinación encontrada en el estudio de hiperparámetros
          sobre el conjunto de split de test, configuración
          usada para generar todos los resultados del explorador de
          secuencias.
        </p>
        <table>
          <thead>
            <tr>
              <th>Algoritmo</th>
              <th>Parámetros</th>
              <th>HOTA(0)</th>
              <th>MOTA</th>
              <th>IDF1</th>
              <th>IDSW</th>
            </tr>
          </thead>
          <tbody>
            {(["bytetracker", "ocsort"] as const).map((algo) => {
              const cfg = data.optimal[algo];
              return (
                <tr key={algo}>
                  <td>{ALGO_LABEL[algo]}</td>
                  <td style={{ textAlign: "left" }}>
                    {Object.entries(cfg.params)
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ")}
                  </td>
                  <td>{cfg.HOTA0.toFixed(2)}</td>
                  <td>{cfg.MOTA.toFixed(2)}</td>
                  <td>{cfg.IDF1.toFixed(2)}</td>
                  <td>{cfg.IDSW}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>Estudio de hiperparámetros</h2>
        <p className="muted">
          Estudio que muestra el efecto de variar cada hiperparámetro por separado sobre HOTA(0)
          en el split de test, manteniendo el resto en su valor óptimo.
        </p>
        <div className="panel-grid">
          {data.hyperparamChartBytetracker && (
            <div>
              <h3>ByteTrack</h3>
              <img
                className="img-frame"
                src={data.hyperparamChartBytetracker}
                alt="Estudio de hiperparámetros de ByteTrack"
              />
            </div>
          )}
          {data.hyperparamChartOcsort && (
            <div>
              <h3>OC-SORT</h3>
              <img
                className="img-frame"
                src={data.hyperparamChartOcsort}
                alt="Estudio de hiperparámetros de OC-SORT"
              />
            </div>
          )}
        </div>
      </div>
    </>
  );
}
