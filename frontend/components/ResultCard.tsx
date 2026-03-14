import { Debate } from "@/types/debate";
import ScoreChart from "./ScoreChart";
import ArgumentGraph from "./ArgumentGraph";
import { Trophy, BarChart3, Brain } from "lucide-react";

interface ResultCardProps {
  debate: Debate;
}

const ResultCard = ({ debate }: ResultCardProps) => {
  if (!debate) {
    return <p>No debate data found</p>;
  }

  const scoreA = debate.analysis.speakerScores.speakerA;
  const scoreB = debate.analysis.speakerScores.speakerB;

  return (
    <div className="max-w-4xl mx-auto mt-10 bg-white border border-gray-200 rounded-xl shadow-md p-8 space-y-8">

      {/* Topic */}
      <div>
        <h1 className="text-2xl font-bold text-gray-800 mb-2">
          {debate.topic}
        </h1>

        <p className="text-gray-600">
          {debate.speakerA} vs {debate.speakerB}
        </p>
      </div>

      {/* Winner */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
        <Trophy className="text-green-600" size={22} />
        <h2 className="text-lg font-semibold text-green-700">
          Winner: {debate.analysis.winner}
        </h2>
      </div>

      {/* Score Chart */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="text-gray-700" size={20} />
          <h3 className="text-lg font-semibold text-gray-800">
            Debate Score Analysis
          </h3>
        </div>

        <ScoreChart
          speakerA={debate.speakerA}
          speakerB={debate.speakerB}
          scoreA={scoreA}
          scoreB={scoreB}
        />
      </div>

      {/* Claims */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Brain className="text-gray-700" size={20} />
          <h3 className="text-lg font-semibold text-gray-800">
            Claims Detected
          </h3>
        </div>

        <ul className="list-disc list-inside text-gray-700 space-y-1">
          {debate.analysis.claims.map((claim, index) => (
            <li key={index}>{claim}</li>
          ))}
        </ul>
      </div>
      <div>
  <h3 className="text-lg font-semibold mb-3">
    Argument Structure
  </h3>

  <ArgumentGraph />
</div>

    </div>
  );
};

export default ResultCard;