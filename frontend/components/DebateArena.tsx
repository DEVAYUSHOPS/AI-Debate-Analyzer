"use client";

import { useRef, useState } from "react";
import { useDebateEngine } from "@/hooks/useDebateEngine";
import { useDebateTimer } from "@/hooks/useDebateTimer";
import SpeechRecorder from "./SpeechRecorder";
import { useRouter } from "next/navigation";



interface Props {
  mode: "text" | "speech";
  topic: string;
  speakerA: string;
  speakerB: string;
}

const DebateArena = ({ mode, topic, speakerA,speakerB }: Props) => {
  const router = useRouter();

  const { round, speaker, nextTurn,isFinished } = useDebateEngine();

  const timeLeft = useDebateTimer(round.time, () => {
    nextTurn();
  });
  const stopRef = useRef<() => void | null>(null);

  const [transcript, setTranscript] = useState({
    opening: { A: "", B: "" },
    rebuttal: { A: "", B: "" },
    closing: { A: "", B: "" }
  });

  const roundKey =
    round.name === "Opening Statement"
      ? "opening"
      : round.name === "Rebuttal"
      ? "rebuttal"
      : "closing";

  const handleTranscriptChange = (
    e: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    const value = e.target.value;

    setTranscript((prev) => ({
      ...prev,
      [roundKey]: {
        ...prev[roundKey],
        [speaker]: value
      }
    }));
  };
  const buildPayload = () => {
  return {
    topic,
    speakerA,
    speakerB,
    mode,
    rounds: [
      {
        round: "Opening Statement",
        speakerA: transcript.opening.A,
        speakerB: transcript.opening.B
      },
      {
        round: "Rebuttal",
        speakerA: transcript.rebuttal.A,
        speakerB: transcript.rebuttal.B
      },
      {
        round: "Closing Statement",
        speakerA: transcript.closing.A,
        speakerB: transcript.closing.B
      }
    ]
  };
};
const submitDebate = async () => {
  try {
    const payload = buildPayload();

    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    console.log("API response:", data);

    router.push(`/result/${data.debateId}`);

  } catch (error) {
    console.error(error);
    alert("Error submitting debate");
  }
  
};

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 space-y-6">
      <div>
  <h1 className="text-lg font-bold text-gray-800">
    {topic}
  </h1>
  <p className="text-gray-600">
    {speakerA} vs {speakerB}
  </p>
</div>

      {/* Round Info */}
      <div className="flex justify-between items-center">

        <div>
          <h2 className="text-xl font-semibold text-gray-800">
            {round.name}
          </h2>

          <p className="text-gray-600">
            Speaker: {speaker}
          </p>
        </div>

        <div className="text-2xl font-bold text-blue-600">
          ⏱ {timeLeft}s
        </div>

      </div>

      {/* Input Area */}

      {mode === "text" && (
        <textarea
          className="w-full border rounded-lg p-4 h-40"
          placeholder="Enter your argument..."
          value={transcript[roundKey][speaker]}
          onChange={handleTranscriptChange}
        />
      )}

      {mode === "speech" && (
        <div className="space-y-4">

          <SpeechRecorder
            onTranscript={(text) => {
              setTranscript((prev) => ({
                ...prev,
                [roundKey]: {
                  ...prev[roundKey],
                  [speaker]: text
                }
              }));
            }}
            onStopRef={(fn) => (stopRef.current = fn)}
          />

          <p className="text-gray-700 border rounded p-3 min-h-20">
            {transcript[roundKey][speaker] || "Transcript will appear here..."}
          </p>

        </div>
      )}

      {/* Controls */}
      <div className="flex justify-end gap-4">

        <button
  onClick={async () => {
    stopRef.current?.();

    if (isFinished) {
      await submitDebate();
      return;
    }

    nextTurn();
  }}
  className="bg-blue-600 text-white px-4 py-2 rounded-md"
>
  {isFinished ? "Finish Debate" : "Next Turn"}
</button>
{isFinished && (
  <p className="text-green-600 font-medium">
    Final turn — ready to submit
  </p>
)}

      </div>

    </div>
  );
};

export default DebateArena;