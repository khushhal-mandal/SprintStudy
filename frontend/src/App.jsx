import { useEffect, useState } from "react";
import Chat from "./Chat";
import Quiz from "./Quiz";
import Upload from "./Upload";
import { listDocuments } from "./api";

// Three views, one useState. No router: there is nothing to deep-link to yet,
// and adding one would be machinery this screen count does not justify.
const TABS = ["upload", "ask", "quiz"];

export default function App() {
  const [tab, setTab] = useState("upload");
  const [doc, setDoc] = useState(null);
  const [indexed, setIndexed] = useState(null);

  // What is already indexed, fetched once on load.
  //
  // Without this the app demands an upload before anything works, which on a
  // deployed instance is a lie - the document is already there. A visitor with
  // no PDF to hand could not try the thing at all.
  useEffect(() => {
    let live = true;
    listDocuments()
      .then((rows) => {
        if (!live) return;
        setIndexed(rows);
        if (rows.length > 0) {
          // The document the API flags as the demo, decided by DEMO_DOCUMENT_ID
          // on the server. Not "newest", which makes whatever a stranger
          // uploaded last into the landing experience, and not "largest",
          // which is only a better guess at the same question. The fallback
          // exists for a local instance where the configured id is absent.
          const landing = rows.find((d) => d.demo) ?? rows[0];
          setDoc(landing);
          setTab("ask");
        }
      })
      .catch(() => live && setIndexed([]));
    return () => {
      live = false;
    };
  }, []);

  const ready = doc?.status === "ready";

  return (
    <main>
      <header>
        <h1>SprintStudy</h1>
        <p className="muted">
          Answers grounded in one document, with the page they came from.
        </p>
      </header>

      <nav>
        {TABS.map((name) => (
          <button
            key={name}
            className={tab === name ? "tab active" : "tab"}
            onClick={() => setTab(name)}
            disabled={name !== "upload" && !ready}
            title={name !== "upload" && !ready ? "Upload a document first" : undefined}
          >
            {name}
          </button>
        ))}
      </nav>

      {/* Which document the answers come from, and a way to change it. Shown on
          every tab because "grounded in one document" is meaningless if the
          screen never says which. */}
      {ready && indexed?.length > 0 && (
        <div className="docbar">
          <span className="muted">Answering from</span>
          <select
            value={doc.document_id}
            onChange={(e) => {
              const next = indexed.find(
                (d) => d.document_id === Number(e.target.value)
              );
              if (next) setDoc(next);
            }}
          >
            {indexed.map((d) => (
              <option key={d.document_id} value={d.document_id}>
                {d.filename} — {d.pages}p, {d.chunk_count} chunks
              </option>
            ))}
          </select>
        </div>
      )}

      {tab === "upload" && (
        <Upload
          doc={doc}
          onReady={(record) => {
            setDoc(record);
            setIndexed((prev) => {
              const rest = (prev ?? []).filter(
                (d) => d.document_id !== record.document_id
              );
              return [record, ...rest];
            });
            setTab("ask");
          }}
        />
      )}
      {tab === "ask" && ready && (
        <Chat key={doc.document_id} documentId={doc.document_id} />
      )}
      {tab === "quiz" && ready && (
        <Quiz key={doc.document_id} documentId={doc.document_id} />
      )}
    </main>
  );
}
