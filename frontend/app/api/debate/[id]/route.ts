import mongoose from "mongoose";
import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import Debate from "@/models/Debate";

export async function GET(
  req: Request,
  context: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await context.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return NextResponse.json(
        { error: "Invalid debate id" },
        { status: 400 }
      );
    }

    await connectDB();

    const debate = await Debate.findById(id);

    if (!debate) {
      return NextResponse.json(
        { error: "Debate not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(debate);
  } catch (error) {
    console.error(error);

    return NextResponse.json(
      { error: "Could not load debate" },
      { status: 500 }
    );
  }
}
