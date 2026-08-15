import { useState } from "react";
import PitchLogo from "./components/PitchLogo";
import ComparativaAlgoritmos from "./components/ComparativaAlgoritmos/ComparativaAlgoritmos";
import ExploradorSecuencias from "./components/ExploradorSecuencias/ExploradorSecuencias";

type Tab = "comparativa" | "explorador";

function App() {
  const [tab, setTab] = useState<Tab>("comparativa");

  return (
    <div className="app">
      <header className="hero">
        <PitchLogo size={72} />
        <div>
          <h1>SoccerNet Tracking Analysis</h1>
          <p>
            Demostración de resultados del pipeline offline de tracking
            multiobjeto sobre SoccerNet-Tracking.
          </p>
        </div>
      </header>

      <nav className="tabs">
        <button
          className={`tab-button ${tab === "comparativa" ? "active" : ""}`}
          onClick={() => setTab("comparativa")}
        >
          Comparativa de algoritmos
        </button>
        <button
          className={`tab-button ${tab === "explorador" ? "active" : ""}`}
          onClick={() => setTab("explorador")}
        >
          Explorador de secuencias
        </button>
      </nav>

      {tab === "comparativa" ? <ComparativaAlgoritmos /> : <ExploradorSecuencias />}
    </div>
  );
}

export default App;
