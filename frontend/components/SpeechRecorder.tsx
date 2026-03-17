"use client";

import { useState, useRef, useEffect } from "react";

interface Props {
  onTranscript: (text: string) => void;
  onStopRef?: (fn: () => void) => void;
}



const SpeechRecorder = ({ onTranscript, onStopRef }: Props) => {

  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<any>(null);

  const startRecording = () => {
    const SpeechRecognition =
      (window as any).webkitSpeechRecognition ||
      (window as any).SpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech Recognition not supported");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onresult = (event: any) => {
      let transcript = "";

      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }

      onTranscript(transcript);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognition.start();

    recognitionRef.current = recognition;
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setIsRecording(false);
  };
  useEffect(() => {
  if (onStopRef) {
    onStopRef(stopRecording);
  }
}, []);
  
  

  return (
    <div className="space-y-4">

      <div className="flex gap-4 justify-center">

        {!isRecording ? (
          <button
            onClick={startRecording}
            className="bg-green-600 text-white px-4 py-2 rounded-md"
          >
            🎤 Start Recording
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="bg-red-500 text-white px-4 py-2 rounded-md"
          >
            ⏹ Stop Recording
          </button>
        )}

      </div>

      {isRecording && (
        <p className="text-center text-gray-600">
          Listening...
        </p>
      )}

    </div>
  );
};

export default SpeechRecorder;