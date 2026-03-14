"use client";

import DebateForm from "@/components/DebateForm";
import PageContainer from "@/components/PageContainer";

const Analyze = () => {
  return (
    <PageContainer>
        <div className="max-w-4xl mx-auto mt-10 px-6">
        

      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800">
          Analyze a Debate
        </h1>

        <p className="text-gray-600 mt-2">
          Enter a debate transcript to detect claims, evaluate argument strength, and determine the winner.
        </p>
      </div>

      {/* Debate Form */}
      <DebateForm />

    </div>
    </PageContainer>
    
  );
};

export default Analyze;