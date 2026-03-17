"use client";

import jsPDF from "jspdf";
import { Debate } from "@/types/debate";

interface Props {
  debate: Debate;
}

const DownloadReportButton = ({ debate }: Props) => {

  const generatePDF = () => {
    const doc = new jsPDF();

    let y = 10;

    // Title
    doc.setFontSize(16);
    doc.text("AI Debate Report", 10, y);
    y += 10;

    // Topic
    doc.setFontSize(12);
    doc.text(`Topic: ${debate.topic}`, 10, y);
    y += 8;

    doc.text(
      `Speakers: ${debate.speakerA} vs ${debate.speakerB}`,
      10,
      y
    );
    y += 8;

    doc.text(`Winner: ${debate.analysis.winner}`, 10, y);
    y += 10;

    // Scores
    doc.text(
      `Scores → ${debate.speakerA}: ${debate.analysis.speakerScores.speakerA}, ${debate.speakerB}: ${debate.analysis.speakerScores.speakerB}`,
      10,
      y
    );
    y += 10;

    // Claims
    doc.text("Claims:", 10, y);
    y += 6;

    debate.analysis.claims.forEach((c) => {
      doc.text(`- ${c}`, 10, y);
      y += 6;
    });

    y += 4;

    // Evidence
    doc.text("Evidence:", 10, y);
    y += 6;

    debate.analysis.evidence?.forEach((e) => {
      doc.text(`- ${e}`, 10, y);
      y += 6;
    });

    y += 4;

    // Transcript
    doc.text("Transcript:", 10, y);
    y += 6;

    debate.rounds.forEach((round) => {
      doc.text(`${round.round}`, 10, y);
      y += 6;

      doc.text(`${debate.speakerA}: ${round.speakerA}`, 10, y);
      y += 6;

      doc.text(`${debate.speakerB}: ${round.speakerB}`, 10, y);
      y += 8;
    });

    doc.save("debate-report.pdf");
  };

  return (
    <button
      onClick={generatePDF}
      className="bg-gray-900 text-white px-4 py-2 rounded-md hover:bg-black flex items-center gap-2"
    >
      📄 Download Report
    </button>
  );
};

export default DownloadReportButton;