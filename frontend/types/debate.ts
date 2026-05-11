export interface Debate {
  _id: string;
  topic: string;
  speakerA: string;
  speakerB: string;
  mode?: "text" | "speech";
  rounds: DebateRound[];

  analysis: {
    winner: string;
    winnerKey?: "speakerA" | "speakerB";

    speakerScores: {
      speakerA: number;
      speakerB: number;
    };

    overallComparison?: string;
    speakerFeedback?: {
      speakerA?: string;
      speakerB?: string;
    };
    finalVerdict?: string;
    feedbackSource?: string;
    feedbackError?: string | null;

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
  cleanedText?: string;
  nlpScore?: number;
  ml: {
    prediction?: {
      argument_quality?: number;
      component?: string;
      stance?: string;
      fallacy?: string;
    };
    rubric_scores?: {
      overall?: number;
      argument_quality?: number;
      evidence_usage?: number;
      logical_reasoning?: number;
      clarity?: number;
      rebuttal_readiness?: number;
    };
    turn_feedback?: {
      recommendation?: string;
      improved_statement?: string;
    };
    student_feedback?: string;
    llm_feedback?: string;
    context?: string;
    retrieval_debug?: unknown;
    feedback_source?: string;
    llm_error?: string | null;
  };
}
