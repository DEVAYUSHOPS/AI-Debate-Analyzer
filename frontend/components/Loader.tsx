const Loader = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[300px] gap-4">

      {/* Spinner */}
      <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>

      {/* Text */}
      <p className="text-gray-600 font-medium">
        Analyzing Debate...
      </p>

    </div>
  );
};

export default Loader;