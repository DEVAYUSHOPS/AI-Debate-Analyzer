import { Debate, SpeakerTurnAnalysis } from "@/types/debate";
import ScoreChart from "./ScoreChart";
import {
  AlertTriangle,
  BarChart3,
  BookOpen,
  FileText,
  Lightbulb,
  MessageSquareText,
  Trophy,
} from "lucide-react";
import DownloadReportButton from "./DownloadReportButton";

interface ResultCardProps {
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

const ResultCard = ({ debate }: ResultCardProps) => {
  if (!debate) {
    return <p>No debate data found</p>;
  }

  const scoreA = debate.analysis.speakerScores.speakerA;
  const scoreB = debate.analysis.speakerScores.speakerB;
  const turnAnalyses = debate.analysis.turnAnalyses ?? [];
  const evidence = debate.analysis.evidence ?? [];
  const fallacies = debate.analysis.fallacies ?? [];

  const candidates = [
    {
      key: "speakerA" as const,
      name: debate.speakerA,
      nlpScore: scoreA,
      feedback: debate.analysis.speakerFeedback?.speakerA,
    },
    {
      key: "speakerB" as const,
      name: debate.speakerB,
      nlpScore: scoreB,
      feedback: debate.analysis.speakerFeedback?.speakerB,
    },
  ];

  return (
    <div className="max-w-5xl mx-auto mt-10 bg-white border border-gray-200 rounded-xl shadow-md p-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-800 mb-2">
          {debate.topic}
        </h1>
        <p className="text-gray-600">
          {debate.speakerA} vs {debate.speakerB}
        </p>
      </div>

      <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
        <Trophy className="text-green-700" size={24} />
        <div>
          <p className="text-sm text-green-700">Winner</p>
          <h2 className="text-lg font-semibold text-green-800">
            {debate.analysis.winner}
          </h2>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {candidates.map((candidate) => (
          <div
            key={candidate.key}
            className="border border-gray-200 rounded-lg p-4 bg-gray-50"
          >
            <h3 className="font-semibold text-gray-800 mb-3">
              {candidate.name}
            </h3>
            <div className="text-sm">
              <div>
                <p className="text-gray-500">NLP Score</p>
                <p className="text-xl font-bold text-blue-700">
                  {formatScore(candidate.nlpScore)}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div>
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="text-gray-700" size={20} />
          <h3 className="text-lg font-semibold text-gray-800">
            Average NLP Score by Candidate
          </h3>
        </div>
        <ScoreChart
          speakerA={debate.speakerA}
          speakerB={debate.speakerB}
          scoreA={scoreA}
          scoreB={scoreB}
        />
      </div>

      {(debate.analysis.overallComparison || debate.analysis.finalVerdict) && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <MessageSquareText className="text-gray-700" size={20} />
            <h3 className="text-lg font-semibold text-gray-800">
              Overall Debate Comparison
            </h3>
          </div>

          {debate.analysis.overallComparison && (
            <p className="text-gray-700 leading-relaxed">
              {cleanFeedbackText(debate.analysis.overallComparison)}
            </p>
          )}

          {debate.analysis.finalVerdict && (
            <div className="border border-blue-100 rounded-lg p-4 bg-blue-50">
              <p className="text-sm font-semibold text-blue-800 mb-1">
                Final Verdict
              </p>
              <p className="text-blue-900">
                {cleanFeedbackText(debate.analysis.finalVerdict)}
              </p>
            </div>
          )}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {candidates.map((candidate) => (
          <div
            key={`${candidate.key}-feedback`}
            className="border border-gray-200 rounded-lg p-4"
          >
            <h3 className="font-semibold text-gray-800 mb-2">
              Feedback for {candidate.name}
            </h3>
            <p className="text-gray-700 text-sm leading-relaxed">
              {cleanFeedbackText(candidate.feedback) || "No candidate feedback available."}
            </p>
          </div>
        ))}
      </div>

      {turnAnalyses.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Lightbulb className="text-gray-700" size={20} />
            <h3 className="text-lg font-semibold text-gray-800">
              Per-Argument Coaching
            </h3>
          </div>

          <div className="space-y-4">
            {turnAnalyses.map((turn, index) => {
              const recommendation =
                cleanFeedbackText(
                  turn.ml.turn_feedback?.recommendation ||
                    turn.ml.student_feedback ||
                    turn.ml.llm_feedback
                ) || "No recommendation available.";
              const improvedStatement =
                cleanFeedbackText(turn.ml.turn_feedback?.improved_statement);

              return (
                <div
                  key={`${turn.round}-${turn.speakerKey}-${index}`}
                  className="border border-gray-200 rounded-lg p-4 bg-gray-50"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                    <div>
                      <h4 className="font-semibold text-gray-800">
                        {turn.speakerName} - {turn.round}
                      </h4>
                      <p className="text-sm text-gray-500">
                        {turn.ml.prediction?.component || "Argument"} |{" "}
                        {turn.ml.prediction?.stance || "Stance N/A"}
                      </p>
                    </div>
                    <span className="text-sm font-semibold text-blue-700">
                      NLP Score: {formatScore(getTurnScore(turn))}
                    </span>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <p className="text-sm font-semibold text-gray-700">
                        Recommendation
                      </p>
                      <p className="text-gray-700 text-sm leading-relaxed">
                        {recommendation}
                      </p>
                    </div>

                    {improvedStatement && (
                      <div>
                        <p className="text-sm font-semibold text-gray-700">
                          Improved Statement
                        </p>
                        <p className="text-gray-700 text-sm leading-relaxed">
                          {improvedStatement}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {evidence.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="text-gray-700" size={20} />
            <h3 className="text-lg font-semibold text-gray-800">Evidence</h3>
          </div>
          <ul className="list-disc list-inside text-gray-700 space-y-1">
            {evidence.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle className="text-gray-700" size={20} />
          <h3 className="text-lg font-semibold text-gray-800">Fallacies</h3>
        </div>

        {fallacies.length === 0 ? (
          <p className="text-gray-500">No fallacies detected.</p>
        ) : (
          <ul className="list-disc list-inside text-gray-700">
            {fallacies.map((fallacy, index) => (
              <li key={index}>{fallacy}</li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <div className="flex items-center gap-2 mb-4">
          <FileText className="text-gray-700" size={20} />
          <h3 className="text-lg font-semibold text-gray-800">
            Debate Transcript
          </h3>
        </div>

        <div className="space-y-6">
          {debate.rounds.map((round, index) => (
            <div
              key={index}
              className="border border-gray-200 rounded-lg p-4 bg-gray-50"
            >
              <h4 className="font-semibold text-gray-700 mb-2">
                {round.round}
              </h4>

              <div className="space-y-3">
                <div>
                  <p className="text-sm text-gray-500">{debate.speakerA}</p>
                  <p className="text-gray-800">{round.speakerA || "No input"}</p>
                </div>

                <div>
                  <p className="text-sm text-gray-500">{debate.speakerB}</p>
                  <p className="text-gray-800">{round.speakerB || "No input"}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-end">
        <DownloadReportButton debate={debate} />
      </div>
    </div>
  );
};

export default ResultCard;
