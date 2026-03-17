"use client";

import { useState } from "react";
import PageContainer from "@/components/PageContainer";
import DebateModeSelector from "@/components/DebateModeSelector";
import DebateArena from "@/components/DebateArena";

const Analyze = () => {

  const [mode, setMode] = useState<"text" | "speech" | null>(null);

const [topic, setTopic] = useState("");
const [speakerA, setSpeakerA] = useState("");
const [speakerB, setSpeakerB] = useState("");

  return (
    <PageContainer>

      <div className="max-w-4xl mx-auto mt-10 px-6">

        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800">
            Analyze a Debate
          </h1>

          <p className="text-gray-600 mt-2">
            Start a structured debate and analyze the arguments automatically.
          </p>
        </div>
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 space-y-6">

  <h2 className="text-xl font-semibold text-gray-800">
    Setup Debate
  </h2>

  {/* Topic */}
  <div>
    <label className="block text-sm text-gray-600 mb-1">
      Topic
    </label>
    <input
      type="text"
      value={topic}
      onChange={(e) => setTopic(e.target.value)}
      placeholder="Enter debate topic"
      className="w-full border rounded-md p-2"
    />
  </div>

  {/* Speaker A */}
  <div>
    <label className="block text-sm text-gray-600 mb-1">
      Speaker A
    </label>
    <input
      type="text"
      value={speakerA}
      onChange={(e) => setSpeakerA(e.target.value)}
      placeholder="Enter name of Speaker A"
      className="w-full border rounded-md p-2"
    />
  </div>

  {/* Speaker B */}
  <div>
    <label className="block text-sm text-gray-600 mb-1">
      Speaker B
    </label>
    <input
      type="text"
      value={speakerB}
      onChange={(e) => setSpeakerB(e.target.value)}
      placeholder="Enter name of Speaker B"
      className="w-full border rounded-md p-2"
    />
  </div>

</div>

        {/* Mode selector appears first */}
        {!mode && (
  <div className="space-y-6 mt-6">

    <DebateModeSelector
      onStart={(selectedMode) => {
        if (!topic || !speakerA || !speakerB) {
          alert("Please fill all fields");
          return;
        }
        setMode(selectedMode);
      }}
    />

  </div>
)}

        {/* Debate arena appears after start */}
        {mode && (
  <DebateArena
    mode={mode}
    topic={topic}
    speakerA={speakerA}
    speakerB={speakerB}
  />
)}

      </div>

    </PageContainer>
  );
};

export default Analyze;