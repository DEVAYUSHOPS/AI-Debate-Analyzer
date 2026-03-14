import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import Debate from "@/models/Debate";

export async function GET(
  req: Request,
) {
  await connectDB();

  const debates = await Debate.find();

  if (!debates) {
    return NextResponse.json(
      { error: "Debate not found" },
      { status: 404 }
    );
  }

  return NextResponse.json(debates);
}