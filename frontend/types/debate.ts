export interface Debate {
  _id: string;
  topic: string;
  speakerA: string;
  speakerB: string;

  analysis: {
    winner: string;

    speakerScores: {
      speakerA: number;
      speakerB: number;
    };

    claims: string[];
    counterclaims: string[];
    evidence: string[];
    fallacies: string[];
    biasLevel: string;
  };
}