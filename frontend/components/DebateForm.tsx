"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Loader from "./Loader";

const DebateForm = () => {
  const router = useRouter();

  const [topic, setTopic] = useState("");
  const [speakerA, setSpeakerA] = useState("");
  const [speakerB, setSpeakerB] = useState("");
  const [transcript, setTranscript] = useState("");
  const [loading, setLoading] = useState(false);

  if (loading) {
    return <Loader />;
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!topic || !speakerA || !speakerB || !transcript) {
      alert("Please fill all fields");
      return;
    }

    try {
      setLoading(true);

      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          topic,
          speakerA,
          speakerB,
          transcript,
        }),
      });

      const data = await res.json();
      router.push(`/result/${data.debateId}`);

    } catch (error) {
      console.log(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white shadow-md rounded-xl p-8 border">

      <h1 className="text-2xl font-bold mb-6 text-gray-800">
        Analyze a Debate
      </h1>

      <form onSubmit={handleSubmit} className="space-y-6">

        {/* Topic */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Topic
          </label>
          <input
            type="text"
            placeholder="Enter debate topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            className="w-full border rounded-md p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        {/* Speaker A */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Speaker A
          </label>
          <input
            type="text"
            placeholder="Enter name of Speaker A"
            value={speakerA}
            onChange={(e) => setSpeakerA(e.target.value)}
            className="w-full border rounded-md p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        {/* Speaker B */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Speaker B
          </label>
          <input
            type="text"
            placeholder="Enter name of Speaker B"
            value={speakerB}
            onChange={(e) => setSpeakerB(e.target.value)}
            className="w-full border rounded-md p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        {/* Transcript */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Debate Transcript
          </label>
          <textarea
            placeholder="Paste debate transcript here..."
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            className="w-full border rounded-md p-3 h-40 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded-md font-medium hover:bg-blue-700 transition"
        >
          Analyze Debate
        </button>

      </form>
    </div>
  );
};

export default DebateForm;