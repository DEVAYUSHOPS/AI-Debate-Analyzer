import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import Debate from "@/models/Debate";

type DebateRound = {
  round: string;
  speakerA?: string;
  speakerB?: string;
};

type MLAnalysisResponse = {
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
  context?: string;
  llm_feedback?: string;
  student_feedback?: string;
  retrieval_debug?: unknown;
};

type SpeakerTurnAnalysis = {
  round: string;
  speakerKey: "speakerA" | "speakerB";
  speakerName: string;
  text: string;
  ml: MLAnalysisResponse;
};

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

const clampScore = (score: number) => Math.max(0, Math.min(10, score));

const getTurnText = (rounds: DebateRound[], speakerKey: "speakerA" | "speakerB") =>
  rounds
    .map((round) => round[speakerKey]?.trim())
    .filter(Boolean)
    .join("\n\n");

const scoreTurn = (analysis: MLAnalysisResponse) => {
  if (typeof analysis.rubric_scores?.overall === "number") {
    return clampScore(analysis.rubric_scores.overall);
  }

  if (typeof analysis.prediction?.argument_quality === "number") {
    return clampScore(analysis.prediction.argument_quality * 10);
  }

  return 0;
};

const compactContext = (context?: string) => {
  if (!context) {
    return null;
  }

  const firstLine = context
    .split("\n")
    .map((line) => line.trim())
    .find(Boolean);

  return firstLine || null;
};

const analyzeTurn = async (
  text: string,
  topic: string,
  studentName: string
): Promise<MLAnalysisResponse> => {
  const response = await fetch(`${ML_SERVICE_URL}/student-feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      topic,
      student_name: studentName,
    }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`ML service returned ${response.status}: ${message}`);
  }

  return response.json();
};

const buildAnalysis = (
  turnAnalyses: SpeakerTurnAnalysis[],
  speakerA: string,
  speakerB: string
) => {
  const speakerATurns = turnAnalyses.filter((turn) => turn.speakerKey === "speakerA");
  const speakerBTurns = turnAnalyses.filter((turn) => turn.speakerKey === "speakerB");

  const averageScore = (turns: SpeakerTurnAnalysis[]) => {
    if (!turns.length) {
      return 0;
    }

    const total = turns.reduce((sum, turn) => sum + scoreTurn(turn.ml), 0);
    return Number((total / turns.length).toFixed(1));
  };

  const speakerAScore = averageScore(speakerATurns);
  const speakerBScore = averageScore(speakerBTurns);

  const claims = turnAnalyses
    .filter((turn) => turn.ml.prediction?.component === "Claim")
    .map((turn) => `${turn.speakerName}: ${turn.text}`);

  const fallacies = turnAnalyses
    .map((turn) => turn.ml.prediction?.fallacy)
    .filter((fallacy): fallacy is string => Boolean(fallacy && fallacy !== "None"));

  const evidence = turnAnalyses
    .map((turn) => compactContext(turn.ml.context))
    .filter((item): item is string => Boolean(item));

  return {
    winner: speakerAScore >= speakerBScore ? speakerA : speakerB,
    speakerScores: {
      speakerA: speakerAScore,
      speakerB: speakerBScore,
    },
    claims: claims.length ? claims : turnAnalyses.map((turn) => `${turn.speakerName}: ${turn.text}`),
    counterclaims: turnAnalyses
      .filter((turn) => turn.ml.prediction?.stance === "CON")
      .map((turn) => `${turn.speakerName}: ${turn.text}`),
    evidence,
    fallacies: [...new Set(fallacies)],
    biasLevel: "N/A",
    turnAnalyses,
  };
};

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { topic, speakerA, speakerB, mode, rounds, transcript } = body;

    const normalizedRounds: DebateRound[] =
      Array.isArray(rounds) && rounds.length
        ? rounds
        : [
            {
              round: "Full Transcript",
              speakerA: transcript,
              speakerB: "",
            },
          ];

    if (!topic || !speakerA || !speakerB || !normalizedRounds.length) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    const speakerAText = getTurnText(normalizedRounds, "speakerA");
    const speakerBText = getTurnText(normalizedRounds, "speakerB");

    if (!speakerAText && !speakerBText) {
      return NextResponse.json(
        { error: "At least one speaker transcript is required" },
        { status: 400 }
      );
    }

    const turnInputs = normalizedRounds
      .flatMap((round) =>
        ([
          {
            speakerKey: "speakerA" as const,
            speakerName: speakerA,
            text: round.speakerA?.trim() || "",
          },
          {
            speakerKey: "speakerB" as const,
            speakerName: speakerB,
            text: round.speakerB?.trim() || "",
          },
        ]).map((turn) => ({
          round: round.round,
          ...turn,
        }))
      )
      .filter((turn) => turn.text);

    const turnAnalyses = await Promise.all(
      turnInputs.map(async (turn) => ({
        ...turn,
        ml: await analyzeTurn(turn.text, topic, turn.speakerName),
      }))
    );

    await connectDB();

    const newDebate = await Debate.create({
      topic,
      speakerA,
      speakerB,
      mode,
      rounds: normalizedRounds,
      analysis: buildAnalysis(turnAnalyses, speakerA, speakerB),
    });

    return NextResponse.json({
      success: true,
      debateId: newDebate._id,
    });
  } catch (error) {
    console.error(error);

    if (error instanceof Error) {
      return NextResponse.json(
        { error: error.message || "Could not reach ML service" },
        { status: 502 }
      );
    }

    return NextResponse.json(
      { error: "Server error" },
      { status: 500 }
    );
  }
}
