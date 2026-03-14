import Link from "next/link";

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto text-center mt-24 px-6">
      
        <h1 className="text-4xl font-bold text-gray-900 mb-6">
          AI Debate Analyzer
        </h1>

      

      <p className="text-gray-600 text-lg mb-10">
        Analyze debates, detect claims, and evaluate argument strength using AI.
      </p>

      <div className="flex justify-center gap-4">

        <Link
          href="/analyze"
          className="bg-blue-600 text-white px-6 py-3 rounded-md font-medium hover:bg-blue-700 transition"
        >
          Start Analyzing
        </Link>

        <Link
          href="/debates"
          className="border border-gray-300 px-6 py-3 rounded-md font-medium hover:bg-gray-100 transition"
        >
          View Debates
        </Link>

      </div>

    </div>
  );
}