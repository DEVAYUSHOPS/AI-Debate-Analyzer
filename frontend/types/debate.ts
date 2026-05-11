export interface Debate {
  _id: string;
  topic: string;
  speakerA: string;
  speakerB: string;
  rounds: DebateRound[];

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
    turnAnalyses?: SpeakerTurnAnalysis[];
  };
}

export interface DebateRound {
  round: string;
  speakerA?: string;
  speakerB?: string;
}

export interface SpeakerTurnAnalysis {
  round: string;
  speakerKey: "speakerA" | "speakerB";
  speakerName: string;
  text: string;
  ml: {
    prediction?: {
      argument_quality?: number;
      component?: string;
      stance?: string;
      fallacy?: string;
    };
    rubric_scores?: {
      overall?: number;
      evidence_usage?: number;
      logical_reasoning?: number;
      clarity?: number;
      rebuttal_readiness?: number;
    };
    student_feedback?: string;
    llm_feedback?: string;
    context?: string;
  };
}
