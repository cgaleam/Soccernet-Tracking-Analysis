export type Algorithm = "bytetracker" | "ocsort";

export interface DistanceSpeedEntry {
  track_id: number;
  distance_m: number;
  avg_speed_kmh: number;
  peak_speed_kmh: number;
}

export interface SequenceEntry {
  sequence: string;
  algorithm: Algorithm;
  playersClassified: number;
  teamCounts: { "0": number; "1": number };
  videoUrl: string;
  heatmapGeneral: string | null;
  heatmapByTeam: string | null;
  heatmapHomography: string | null;
  topDistanceSpeed: DistanceSpeedEntry[] | null;
  possessionPct: Record<string, number> | null;
  videoFrameRange: [number, number];
}

export interface SplitMetrics {
  HOTA: number;
  HOTA0: number;
  DetA: number;
  AssA: number;
  MOTA: number;
  IDF1: number;
  IDSW: number;
}

export interface OptimalConfig {
  params: Record<string, number>;
  HOTA0: number;
  MOTA: number;
  IDF1: number;
  IDSW: number;
}

export interface HyperparamRow {
  algo: Algorithm;
  tracker: string;
  param: string;
  value: string;
  HOTA0: number;
  MOTA: number;
  IDF1: number;
  IDSW: number;
}

export interface GlobalMetrics {
  bySplit: {
    train: Record<Algorithm, SplitMetrics>;
    test: Record<Algorithm, SplitMetrics>;
  };
  optimal: Record<Algorithm, OptimalConfig>;
  hyperparameterStudy: HyperparamRow[];
  hotaVsHota0Chart: string | null;
  hyperparamChartBytetracker: string | null;
  hyperparamChartOcsort: string | null;
}
