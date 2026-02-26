import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import Debate from "@/models/Debate";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { topic, speakerA, speakerB, transcript } = body;

    // Basic validation
    if (!topic || !speakerA || !speakerB || !transcript) {
      return NextResponse.json(
        { error: "All fields are required" },
        { status: 400 }
      );
    }

    await connectDB();

    // 🔥 Dummy ML Response (Replace later)
    const dummyAnalysis = {
      claims: ["AI improves efficiency"],
      counterclaims: ["AI lacks emotional intelligence"],
      evidence: ["Studies show 30% productivity gain"],
      fallacies: [],
      biasLevel: "Low",
      speakerScores: {
        speakerA: 8.2,
        speakerB: 6.7
      },
      winner: speakerA
    };

    const newDebate = await Debate.create({
      topic,
      speakerA,
      speakerB,
      transcript,
      analysis: dummyAnalysis
    });

    return NextResponse.json({
      success: true,
      debateId: newDebate._id
    });

  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { error: "Server error" },
      { status: 500 }
    );
  }
}