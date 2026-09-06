import { useState } from "react";
import Answer from "./Citation";
import { ask } from "./api";

export default function Chat({ documentId }) {
  const [turns, setTurns] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);

  async function send(event) {
    event.preventDefault();
    const asked = question.trim();
    if (!asked || busy) return;

    setQuestion("");
    setBusy(true);
    try {
      const result = await ask(asked, documentId);
      setTurns((prev) => [...prev, { question: asked, result }]);
    } catch (err) {
      // 503 carries a detail naming the cause - a rate limit, a stopped
      // container. Showing it beats "something went wrong".
      setTurns((prev) => [...prev, { question: asked, error: err.detail, status: err.status }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h2>Ask</h2>
      <form onSubmit={send} className="row">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask something the document covers…"
          disabled={busy}
        />
        <button disabled={busy || !question.trim()}>{busy ? "…" : "Ask"}</button>
      </form>

      {turns.length === 0 && (
        <p className="muted">
          Answers are grounded in the document only. Anything it does not cover is refused
          rather than guessed at.
        </p>
      )}

      {turns.map((turn, i) => (
        <div key={i} className="card">
          <p className="question">{turn.question}</p>

          {turn.error && (
            <p className="error">
              {turn.status === 409 ? "Document is still being processed. " : ""}
              {turn.error}
            </p>
          )}

          {turn.result && !turn.result.answered && (
            <p className="refused">
              Not covered in this document.
              <span className="muted">
                {" "}
                ({turn.result.refusal_reason === "pre_filter"
                  ? "nothing relevant retrieved"
                  : "the model found no answer in the retrieved passages"})
              </span>
            </p>
          )}

          {turn.result?.answered && (
            <>
              <Answer text={turn.result.answer} citations={turn.result.citations} />
              {turn.result.uncited && (
                <p className="warn">
                  The model did not cite a passage for this answer. Treat it with more
                  caution than a cited one.
                </p>
              )}
              {turn.result.dropped_citations?.length > 0 && (
                <p className="warn">
                  Dropped markers pointing at no passage:{" "}
                  {turn.result.dropped_citations.join(", ")}
                </p>
              )}
            </>
          )}
        </div>
      ))}
    </section>
  );
}
