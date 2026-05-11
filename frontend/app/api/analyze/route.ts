import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import Debate from "@/models/Debate";

type DebateRound = {
  round: string;
  speakerA?: string;
  speakerB?: string;
};

type MLDebateFeedbackResponse = {
  analysis?: Record<string, unknown>;
  error?: string;
  detail?: string | { msg?: string }[];
};

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

const normalizeRounds = (rounds: unknown, transcript?: string): DebateRound[] => {
  if (Array.isArray(rounds) && rounds.length) {
    return rounds
      .map((round) => {
        const item = round as Partial<DebateRound>;
        return {
          round: item.round || "Debate Round",
          speakerA: item.speakerA || "",
          speakerB: item.speakerB || "",
        };
      })
      .filter((round) => round.speakerA?.trim() || round.speakerB?.trim());
  }

  if (transcript?.trim()) {
    return [
      {
        round: "Full Transcript",
        speakerA: transcript,
        speakerB: "",
      },
    ];
  }

  return [];
};

const readMLError = (data: MLDebateFeedbackResponse) => {
  if (typeof data.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data.detail)) {
    return data.detail.map((item) => item.msg).filter(Boolean).join(", ");
  }

  return data.error || "Debate analysis failed";
};

const analyzeDebate = async (payload: {
  topic: string;
  speakerA: string;
  speakerB: string;
  mode: "text" | "speech";
  rounds: DebateRound[];
}) => {
  const response = await fetch(`${ML_SERVICE_URL}/debate-feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = (await response.json()) as MLDebateFeedbackResponse;

  if (!response.ok || !data.analysis) {
    throw new Error(readMLError(data));
  }

  return data.analysis;
};

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { topic, speakerA, speakerB, mode, rounds, transcript } = body;
    const normalizedRounds = normalizeRounds(rounds, transcript);

    if (!topic || !speakerA || !speakerB || !normalizedRounds.length) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    const normalizedMode: "text" | "speech" = mode === "speech" ? "speech" : "text";
    const analysis = await analyzeDebate({
      topic,
      speakerA,
      speakerB,
      mode: normalizedMode,
      rounds: normalizedRounds,
    });

    await connectDB();

    const newDebate = await Debate.create({
      topic,
      speakerA,
      speakerB,
      mode: normalizedMode,
      rounds: normalizedRounds,
      analysis,
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
