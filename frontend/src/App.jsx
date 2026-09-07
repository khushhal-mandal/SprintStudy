import { useState } from "react";
import Chat from "./Chat";
import Quiz from "./Quiz";
import Upload from "./Upload";

// Three views, one useState. No router: there is nothing to deep-link to yet,
// and adding one would be machinery this screen count does not justify.
const TABS = ["upload", "ask", "quiz"];

export default function App() {
  const [tab, setTab] = useState("upload");
  const [doc, setDoc] = useState(null);

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

      {tab === "upload" && (
        <Upload
          doc={doc}
          onReady={(record) => {
            setDoc(record);
            setTab("ask");
          }}
        />
      )}
      {tab === "ask" && ready && <Chat documentId={doc.document_id} />}
      {tab === "quiz" && ready && <Quiz documentId={doc.document_id} />}
    </main>
  );
}
