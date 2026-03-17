"use client";

import { useState } from "react";

interface Props {
  onStart: (mode: "text" | "speech") => void;
}

const DebateModeSelector = ({ onStart }: Props) => {
  const [mode, setMode] = useState<"text" | "speech">("text");

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 space-y-6">

      <h2 className="text-xl font-semibold text-gray-800">
        Choose Debate Mode
      </h2>

      <div className="flex gap-6">

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            value="text"
            checked={mode === "text"}
            onChange={() => setMode("text")}
          />
          Text Debate
        </label>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            value="speech"
            checked={mode === "speech"}
            onChange={() => setMode("speech")}
          />
          Speech Debate
        </label>

      </div>

      <button
        onClick={() => onStart(mode)}
        className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition"
      >
        Start Debate
      </button>

    </div>
  );
};

export default DebateModeSelector;