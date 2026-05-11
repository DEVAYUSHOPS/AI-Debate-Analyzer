import { useEffect, useRef, useState } from "react";

export const useDebateTimer = (
  initialTime: number,
  onFinish: () => void,
  resetKey: string
) => {
  const [timeLeft, setTimeLeft] = useState(initialTime);
  const onFinishRef = useRef(onFinish);
  const hasFinishedRef = useRef(false);

  useEffect(() => {
    onFinishRef.current = onFinish;
  }, [onFinish]);

  useEffect(() => {
    hasFinishedRef.current = false;
    // The timer must reset when a new debate round starts.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTimeLeft(initialTime);
  }, [initialTime, resetKey]);

  useEffect(() => {
    if (timeLeft <= 0) {
      if (!hasFinishedRef.current) {
        hasFinishedRef.current = true;
        onFinishRef.current();
      }

      return;
    }

    const timer = window.setTimeout(() => {
      setTimeLeft((prev) => Math.max(prev - 1, 0));
    }, 1000);

    return () => window.clearTimeout(timer);
  }, [timeLeft]);

  return timeLeft;
};
