import mongoose from "mongoose";

const RoundSchema = new mongoose.Schema({
  round: String,
  speakerA: String,
  speakerB: String
});

const DebateSchema = new mongoose.Schema({
  topic: {
    type: String,
    default: "Untitled Debate"
  },

  speakerA: String,
  speakerB: String,

  mode: {
    type: String,
    enum: ["text", "speech"]
  },

  rounds: [RoundSchema],

  analysis: {
    winner: String,
    speakerScores: {
      speakerA: Number,
      speakerB: Number
    },
    claims: [String],
    evidence: [String],
    fallacies: [String]
  }
}, { timestamps: true });

export default mongoose.models.Debate || mongoose.model("Debate", DebateSchema);