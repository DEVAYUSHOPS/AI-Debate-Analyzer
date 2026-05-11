"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useDebateEngine } from "@/hooks/useDebateEngine";
import { useDebateTimer } from "@/hooks/useDebateTimer";
import SpeechRecorder from "./SpeechRecorder";

interface Props {
  mode: "text" | "speech";
  topic: string;
  speakerA: string;
  speakerB: string;
}

const DebateArena = ({ mode, topic, speakerA, speakerB }: Props) => {
  const router = useRouter();
  const { round, speaker, nextTurn, isFinished } = useDebateEngine();
  const stopRef = useRef<(() => void) | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const timeLeft = useDebateTimer(
    round.time,
    () => {
      if (!isFinished) {
        nextTurn();
      }
    },
    `${round.name}-${speaker}`
  );

  const [transcript, setTranscript] = useState({
    opening: { A: "", B: "" },
    rebuttal: { A: "", B: "" },
    closing: { A: "", B: "" },
  });

  const roundKey =
    round.name === "Opening Statement"
      ? "opening"
      : round.name === "Rebuttal"
      ? "rebuttal"
      : "closing";

  const currentSpeakerName = speaker === "A" ? speakerA : speakerB;
  const currentTurnText = transcript[roundKey][speaker];

  const handleTranscriptChange = (
    e: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    const value = e.target.value;

    setTranscript((prev) => ({
      ...prev,
      [roundKey]: {
        ...prev[roundKey],
        [speaker]: value,
      },
    }));
  };

  const buildPayload = () => ({
    topic,
    speakerA,
    speakerB,
    mode,
    rounds: [
      {
        round: "Opening Statement",
        speakerA: transcript.opening.A,
        speakerB: transcript.opening.B,
      },
      {
        round: "Rebuttal",
        speakerA: transcript.rebuttal.A,
        speakerB: transcript.rebuttal.B,
      },
      {
        round: "Closing Statement",
        speakerA: transcript.closing.A,
        speakerB: transcript.closing.B,
      },
    ],
  });

  const submitDebate = async () => {
    try {
      setIsSubmitting(true);

      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(buildPayload()),
      });

      const data = await res.json();

      if (!res.ok || !data.debateId) {
        throw new Error(data.error || "Debate analysis failed");
      }

      router.push(`/result/${data.debateId}`);
    } catch (error) {
      console.error(error);
      alert(error instanceof Error ? error.message : "Error submitting debate");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 space-y-6">
      <div>
        <h1 className="text-lg font-bold text-gray-800">{topic}</h1>
        <p className="text-gray-600">
          {speakerA} vs {speakerB}
        </p>
      </div>

      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">{round.name}</h2>
          <p className="text-gray-600">Speaker: {currentSpeakerName}</p>
        </div>

        <div className="text-2xl font-bold text-blue-600">
          Timer: {timeLeft}s
        </div>
      </div>

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
                  [speaker]: text,
                },
              }));
            }}
            onStopRef={(fn) => (stopRef.current = fn)}
          />

          <p className="text-gray-700 border rounded p-3 min-h-20">
            {transcript[roundKey][speaker] || "Transcript will appear here..."}
          </p>
        </div>
      )}

      <div className="flex justify-end gap-4">
        <button
          onClick={async () => {
            stopRef.current?.();

            if (!currentTurnText.trim()) {
              alert(`Please enter ${currentSpeakerName}'s argument before continuing.`);
              return;
            }

            if (isFinished) {
              await submitDebate();
              return;
            }

            nextTurn();
          }}
          disabled={isSubmitting}
          className="bg-blue-600 text-white px-4 py-2 rounded-md disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {isSubmitting ? "Analyzing..." : isFinished ? "Finish Debate" : "Next Turn"}
        </button>

        {isFinished && (
          <p className="text-green-600 font-medium">
            Final turn - ready to submit
          </p>
        )}
      </div>
    </div>
  );
};

export default DebateArena;
