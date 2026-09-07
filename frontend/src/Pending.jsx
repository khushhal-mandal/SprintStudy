import { useEffect, useState } from "react";

// A spinner that admits how long it has been going.
//
// The API is on a free instance that sleeps after ~15 minutes idle, so the
// first request after a quiet period spends ~30s waking it before any work
// starts. A bare "…" for thirty seconds is indistinguishable from a hang, and
// the honest fix is to say what is happening rather than to hide it.
//
// The threshold is 4s because a warm request finishes in 5-14s: below that the
// message would fire on every normal call and become noise.
const COLD_START_HINT_AFTER = 4;

export default function Pending({ label }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <p className="pending" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      {label} <span className="elapsed">{seconds}s</span>
      {seconds >= COLD_START_HINT_AFTER && (
        <span className="muted">
          {" "}
          — the API sleeps when idle, so the first request takes about 30s to
          wake it. Later ones are fast.
        </span>
      )}
    </p>
  );
}
