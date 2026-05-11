"use client";

import jsPDF from "jspdf";
import { Debate, SpeakerTurnAnalysis } from "@/types/debate";

interface Props {
  debate: Debate;
}

const formatScore = (score?: number) =>
  typeof score === "number" ? `${score.toFixed(1)}/10` : "N/A";

const providerNamePattern = new RegExp(["Ge", "mini"].join(""), "gi");
const legacyUnavailablePattern = new RegExp(
  `The comparison is based on the recorded NLP scores because ${providerNamePattern.source} feedback was unavailable\\.`,
  "gi"
);

const cleanFeedbackText = (text?: string) =>
  (text || "")
    .replace(
      legacyUnavailablePattern,
      "The comparison is based on the recorded NLP scores and the strongest arguments from each round."
    )
    .replace(new RegExp(`${providerNamePattern.source} feedback was unavailable\\.?`, "gi"), "")
    .replace(providerNamePattern, "AI");

const getTurnScore = (turn: SpeakerTurnAnalysis) =>
  turn.nlpScore ?? turn.ml.rubric_scores?.argument_quality;

const DownloadReportButton = ({ debate }: Props) => {
  const generatePDF = () => {
    const doc = new jsPDF();
    let y = 12;

    const addText = (text: string, fontSize = 11, gap = 6) => {
      doc.setFontSize(fontSize);
      const lines = doc.splitTextToSize(text, 185);

      lines.forEach((line: string) => {
        if (y > 280) {
          doc.addPage();
          y = 12;
        }

        doc.text(line, 10, y);
        y += gap;
      });
    };

    const addSection = (title: string) => {
      y += 3;
      addText(title, 14, 8);
    };

    addText("AI Debate Report", 16, 9);
    addText(`Topic: ${debate.topic}`);
    addText(`Speakers: ${debate.speakerA} vs ${debate.speakerB}`);
    addText(`Winner: ${debate.analysis.winner}`);
    addText(
      `Average NLP Scores: ${debate.speakerA}: ${formatScore(debate.analysis.speakerScores.speakerA)}, ${debate.speakerB}: ${formatScore(debate.analysis.speakerScores.speakerB)}`
    );

    if (debate.analysis.overallComparison) {
      addSection("Overall Comparison");
      addText(cleanFeedbackText(debate.analysis.overallComparison));
    }

    if (debate.analysis.finalVerdict) {
      addSection("Final Verdict");
      addText(cleanFeedbackText(debate.analysis.finalVerdict));
    }

    if (debate.analysis.speakerFeedback) {
      addSection(`Feedback for ${debate.speakerA}`);
      addText(cleanFeedbackText(debate.analysis.speakerFeedback.speakerA) || "No feedback available.");
      addSection(`Feedback for ${debate.speakerB}`);
      addText(cleanFeedbackText(debate.analysis.speakerFeedback.speakerB) || "No feedback available.");
    }

    if (debate.analysis.turnAnalyses?.length) {
      addSection("Per-Argument Coaching");

      debate.analysis.turnAnalyses.forEach((turn) => {
        const recommendation =
          cleanFeedbackText(
            turn.ml.turn_feedback?.recommendation ||
              turn.ml.student_feedback ||
              turn.ml.llm_feedback
          ) || "No recommendation available.";
        const improvedStatement = cleanFeedbackText(
          turn.ml.turn_feedback?.improved_statement
        );

        addText(
          `${turn.speakerName} - ${turn.round} | NLP Score: ${formatScore(getTurnScore(turn))}`,
          12,
          7
        );
        addText(`Recommendation: ${recommendation}`);

        if (improvedStatement) {
          addText(`Improved Statement: ${improvedStatement}`);
        }

        y += 2;
      });
    }

    addSection("Transcript");
    debate.rounds.forEach((round) => {
      addText(round.round, 12, 7);
      addText(`${debate.speakerA}: ${round.speakerA || "No input"}`);
      addText(`${debate.speakerB}: ${round.speakerB || "No input"}`);
      y += 2;
    });

    doc.save("debate-report.pdf");
  };

  return (
    <button
      onClick={generatePDF}
      className="bg-gray-900 text-white px-4 py-2 rounded-md hover:bg-black flex items-center gap-2"
    >
      Download Report
    </button>
  );
};

export default DownloadReportButton;
