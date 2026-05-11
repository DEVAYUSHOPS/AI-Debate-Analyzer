import mongoose from "mongoose";

const { Schema } = mongoose;

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
    winnerKey: String,
    speakerScores: {
      speakerA: Number,
      speakerB: Number
    },
    overallComparison: String,
    speakerFeedback: {
      speakerA: String,
      speakerB: String
    },
    finalVerdict: String,
    feedbackSource: String,
    feedbackError: String,
    claims: [String],
    counterclaims: [String],
    evidence: [String],
    fallacies: [String],
    biasLevel: String,
    turnAnalyses: [Schema.Types.Mixed]
  }
}, { timestamps: true });

export default mongoose.models.Debate || mongoose.model("Debate", DebateSchema);
