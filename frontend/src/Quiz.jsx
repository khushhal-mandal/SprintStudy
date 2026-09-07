import { useState } from "react";
import { generateQuiz, submitQuiz } from "./api";

export default function Quiz({ documentId }) {
  const [range, setRange] = useState({ startPage: 93, endPage: 103, n: 5 });
  const [quiz, setQuiz] = useState(null);
  const [choices, setChoices] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function generate() {
    setBusy(true);
    setError(null);
    setResult(null);
    setChoices({});
    try {
      setQuiz(await generateQuiz({ ...range, documentId }));
    } catch (err) {
      setError(
        err.status === 422
          ? `No usable chunks on pages ${range.startPage}-${range.endPage}. Try another range.`
          : err.detail
      );
      setQuiz(null);
    } finally {
      setBusy(false);
    }
  }

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const responses = quiz.questions.map((q) => choices[q.position] ?? -1);
      setResult(await submitQuiz(quiz.quiz_id, responses));
    } catch (err) {
      setError(err.detail);
    } finally {
      setBusy(false);
    }
  }

  const answered = quiz ? quiz.questions.every((q) => choices[q.position] !== undefined) : false;
  const markFor = (position) => result?.marks.find((m) => m.position === position);

  return (
    <section>
      <h2>Quiz</h2>

      <div className="row">
        <label>
          pages{" "}
          <input
            type="number"
            value={range.startPage}
            onChange={(e) => setRange({ ...range, startPage: Number(e.target.value) })}
          />
        </label>
        <label>
          to{" "}
          <input
            type="number"
            value={range.endPage}
            onChange={(e) => setRange({ ...range, endPage: Number(e.target.value) })}
          />
        </label>
        <label>
          questions{" "}
          <input
            type="number"
            value={range.n}
            onChange={(e) => setRange({ ...range, n: Number(e.target.value) })}
          />
        </label>
        <button onClick={generate} disabled={busy}>
          {busy ? "…" : "Generate"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="card score">
          <strong>
            {result.score} / {result.total}
          </strong>
        </div>
      )}

      {quiz?.questions.map((q) => {
        const mark = markFor(q.position);
        return (
          <div key={q.position} className="card">
            <p className="question">
              {q.position + 1}. {q.question}
            </p>
            <ul className="options">
              {q.options.map((option, index) => {
                const chosen = choices[q.position] === index;
                let state = "";
                if (mark) {
                  if (index === mark.correct_index) state = "correct";
                  else if (index === mark.submitted) state = "wrong";
                }
                return (
                  <li key={index}>
                    <label className={state}>
                      <input
                        type="radio"
                        name={`q${q.position}`}
                        checked={chosen}
                        disabled={Boolean(result)}
                        onChange={() => setChoices({ ...choices, [q.position]: index })}
                      />
                      {String.fromCharCode(65 + index)}. {option}
                    </label>
                  </li>
                );
              })}
            </ul>
            {mark && (
              <p className="feedback">
                <span className={mark.correct ? "correct" : "wrong"}>
                  {mark.correct ? "Correct" : "Incorrect"}
                </span>{" "}
                <span className="pill">p.{mark.page}</span> {mark.explanation}
              </p>
            )}
          </div>
        );
      })}

      {quiz && !result && (
        <button onClick={submit} disabled={!answered || busy}>
          {answered ? "Submit" : "Answer every question first"}
        </button>
      )}
    </section>
  );
}
