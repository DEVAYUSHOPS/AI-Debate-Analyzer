// models/Debate.ts

import mongoose from "mongoose";

const DebateSchema = new mongoose.Schema({
  topic: {
    type: String,
    required: true
  },

  speakerA: {
    type: String,
    required: true
  },

  speakerB: {
    type: String,
    required: true
  },

  transcript: {
    type: String,
    required: true
  },

  analysis: {
    claims: Array,
    counterclaims: Array,
    evidence: Array,
    fallacies: Array,
    biasLevel: String,
    speakerScores: {
      speakerA: Number,
      speakerB: Number
    },
    winner: String
  },

  createdAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.models.Debate || mongoose.model("Debate", DebateSchema);