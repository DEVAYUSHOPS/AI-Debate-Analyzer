"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

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
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer>
        <BarChart data={data}>
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="score" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ScoreChart;