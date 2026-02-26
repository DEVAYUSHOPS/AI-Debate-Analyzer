import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import Debate from "@/models/Debate";

export async function GET(
  req: Request,
  context: { params: Promise<{ id: string }> }
) {
  await connectDB();

  const { id } = await context.params;  // ✅ UNWRAP HERE

  const debate = await Debate.findById(id);

  if (!debate) {
    return NextResponse.json(
      { error: "Debate not found" },
      { status: 404 }
    );
  }

  return NextResponse.json(debate);
}