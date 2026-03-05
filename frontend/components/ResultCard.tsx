import { Debate } from "@/types/debate";
import ScoreChart from './ScoreChart';

interface ResultCardProps {
  debate: Debate;
}

const ResultCard = ({ debate } : ResultCardProps) => {
    if(!debate){
        return <p>No debate data found</p>
    }

    console.log(debate);
    const scoreA = debate.analysis.speakerScores.speakerA;
    const scoreB = debate.analysis.speakerScores.speakerB;

  return (
     <div>

      <h2>{debate.topic}</h2>

      <h3>
        {debate.speakerA} vs {debate.speakerB}
      </h3>

      <h2>Winner: {debate.analysis.winner}</h2>

      <ScoreChart
        speakerA={debate.speakerA}
        speakerB={debate.speakerB}
        scoreA={scoreA}
        scoreB={scoreB}
      />

    </div>
  )
}

export default ResultCard