import { Debate } from "@/types/debate";
import Link from "next/link";

interface AllDebatesProps {
  debates: Debate[];
}

const AllDebates = ({ debates }: AllDebatesProps) => {

  return (
    <div className="max-w-4xl mx-auto mt-10 space-y-6">

      <h1 className="text-2xl font-bold text-gray-800">
        Past Debates
      </h1>

      {debates.map((debate: Debate) => (
        <div
          key={debate._id}
          className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm hover:shadow-md transition"
        >

          {/* Topic */}
          <h2 className="text-lg font-semibold text-gray-800 mb-2">
            {debate.topic}
          </h2>

          {/* Speakers */}
          <p className="text-gray-600 mb-4">
            {debate.speakerA} vs {debate.speakerB}
          </p>

          {/* Action */}
          <Link
            href={`/result/${debate._id}`}
            className="inline-block bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition"
          >
            View Result
          </Link>

        </div>
      ))}

    </div>
  );
};

export default AllDebates;