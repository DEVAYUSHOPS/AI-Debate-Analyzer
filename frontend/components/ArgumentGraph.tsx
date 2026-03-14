"use client";

import React from "react";
import ReactFlow from "reactflow";
import "reactflow/dist/style.css";

const nodes = [
  { id: "1", position: { x: 250, y: 0 }, data: { label: "Claim" } },
  { id: "2", position: { x: 100, y: 100 }, data: { label: "Evidence" } },
  { id: "3", position: { x: 400, y: 100 }, data: { label: "Counterclaim" } },
];

const edges = [
  { id: "e1", source: "2", target: "1" },
  { id: "e2", source: "3", target: "1" },
];

const ArgumentGraph = () => {
  return (
    <div style={{ height: 300 }}>
      <ReactFlow nodes={nodes} edges={edges} fitView />
    </div>
  );
};

export default ArgumentGraph;