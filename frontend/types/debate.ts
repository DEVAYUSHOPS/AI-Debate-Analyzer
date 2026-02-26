export interface AnalysisResult {
  claims: any[];
  counterclaims: any[];
  evidence: any[];
  fallacies: any[];
  biasLevel: string;
  speakerScores: {
    speakerA: number;
    speakerB: number;
  };
  winner: string;
}