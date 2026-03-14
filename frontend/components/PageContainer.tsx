import React from "react";

const PageContainer = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      {children}
    </div>
  );
};

export default PageContainer;