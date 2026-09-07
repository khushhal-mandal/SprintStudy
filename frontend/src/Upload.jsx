import { useEffect, useRef, useState } from "react";
import Pending from "./Pending";
import { getDocument, uploadPdf } from "./api";

const SETTLED = ["ready", "failed"];

export default function Upload({ doc, onReady }) {
  const [status, setStatus] = useState(doc);

  // Held in a ref so the polling effect depends on status alone. App passes an
  // inline arrow, whose identity changes every render - in the dependency array
  // that tears down and restarts the interval on each one.
  const ready = useRef(onReady);
  ready.current = onReady;
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  // Ingestion runs in the background - a 296-page PDF takes minutes - so the
  // only way to observe it is to poll until the row settles.
  useEffect(() => {
    if (!status || SETTLED.includes(status.status)) return undefined;
    const timer = setInterval(async () => {
      try {
        const next = await getDocument(status.document_id);
        setStatus(next);
        if (next.status === "ready") ready.current(next);
      } catch (err) {
        setError(err.detail);
        clearInterval(timer);
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [status]);

  async function handle(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const started = await uploadPdf(file);
      // Same bytes as an earlier upload: already indexed, nothing to poll.
      const record = started.reused
        ? await getDocument(started.document_id)
        : { ...started, filename: file.name };
      setStatus(record);
      if (record.status === "ready") ready.current(record);
    } catch (err) {
      setError(err.detail);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h2>Upload a PDF</h2>
      <input type="file" accept="application/pdf" onChange={handle} disabled={busy} />
      {busy && <Pending label="Uploading…" />}

      {status && (
        <div className="card">
          <div className="row">
            <strong>{status.filename ?? "document"}</strong>
            <span className={`status ${status.status}`}>{status.status}</span>
          </div>
          {!SETTLED.includes(status.status) && (
            <p className="muted">
              Parsing, chunking and embedding. This takes a few minutes for a long PDF.
            </p>
          )}
          {status.status === "ready" && (
            <p className="muted">
              {status.pages} pages, {status.chunk_count} chunks indexed.
            </p>
          )}
          {status.status === "failed" && <p className="error">{status.error}</p>}
        </div>
      )}

      {error && <p className="error">{error}</p>}
    </section>
  );
}
