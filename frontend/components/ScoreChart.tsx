"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid
} from "recharts";

interface ScoreChartProps {
  speakerA: string;
  speakerB: string;
  scoreA: number;
  scoreB: number;
}

const ScoreChart = ({ speakerA, speakerB, scoreA, scoreB }: ScoreChartProps) => {

  const data = [
    {
      name: speakerA,
      score: scoreA
    },
    {
      name: speakerB,
      score: scoreB
    }
  ];

  return (
    <div className="w-full h-[300px] bg-gray-50 border border-gray-200 rounded-lg p-4">

      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>

          {/* Grid */}
          <CartesianGrid strokeDasharray="3 3" />

          {/* X Axis */}
          <XAxis
            dataKey="name"
            tick={{ fill: "#374151", fontSize: 14 }}
          />

          {/* Y Axis */}
          <YAxis
            tick={{ fill: "#374151", fontSize: 14 }}
            domain={[0, 10]}
          />

          {/* Tooltip */}
          <Tooltip
            contentStyle={{
              borderRadius: "8px",
              border: "1px solid #e5e7eb"
            }}
          />

          {/* Bars */}
          <Bar
            dataKey="score"
            fill="#2563eb"
            radius={[6, 6, 0, 0]}
          />

        </BarChart>
      </ResponsiveContainer>

    </div>
  );
};

export default ScoreChart;