"use client";

import Link from "next/link";

const Navbar = () => {
  return (
    <nav className="border-b bg-white">
      <div className="max-w-5xl mx-auto px-6 py-4 flex justify-between items-center">

        {/* Logo / Title */}
        <h1 className="text-xl font-bold text-gray-800">
          <Link href="/" className="hover:text-blue-600 transition">
    AI Debate Analyzer
  </Link>
        </h1>

        {/* Navigation Links */}
        <div className="flex gap-6 text-gray-600 font-medium">
          <Link href="/analyze" className="hover:text-black">
            Analyze
          </Link>

          <Link href="/debates" className="hover:text-black">
            Debates
          </Link>
        </div>

      </div>
    </nav>
  );
};

export default Navbar;