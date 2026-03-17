import { useState } from "react";
import { debateRounds } from "@/lib/debateConfig";

export const useDebateEngine = () => {

  const [roundIndex, setRoundIndex] = useState(0);
  const [speaker, setSpeaker] = useState<"A" | "B">("A");

  const isFinished =
    roundIndex === debateRounds.length - 1 && speaker === "B";

  const nextTurn = () => {
    if (speaker === "A") {
      setSpeaker("B");
    } else {
      setSpeaker("A");
      setRoundIndex((prev) =>
  Math.min(prev + 1, debateRounds.length - 1)
);
    }
  };

  return {
    round: debateRounds[roundIndex],
    roundIndex,
    speaker,
    nextTurn,
    isFinished
  };
};