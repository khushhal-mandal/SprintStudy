// Fetch wrappers. One error shape, so every view surfaces the same thing.
//
// The API reports upstream failures as 503 with a `detail` naming the cause -
// a stopped container, an exhausted token quota. Swallowing that in favour of a
// generic message would throw away the only useful part of the response.

import fixtures from "./fixtures/quiz.json";

// /quiz/generate has never returned a real LLM response - the Groq daily quota
// ran out before it could be exercised. Flip this to false once quota returns,
// and replace the fixture with a captured real response.
export const USE_FIXTURE = true;

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`/api${path}`, options);
  } catch (cause) {
    throw new ApiError(0, `Cannot reach the API. Is uvicorn running? (${cause.message})`);
  }

  const body = await response.text();
  let parsed = null;
  try {
    parsed = body ? JSON.parse(body) : null;
  } catch {
    // Non-JSON body: a proxy error page, or FastAPI's bare "Internal Server Error".
  }

  if (!response.ok) {
    const detail = parsed?.detail ?? body ?? response.statusText;
    throw new ApiError(response.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return parsed;
}

export function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/upload", { method: "POST", body: form });
}

export function getDocument(id) {
  return request(`/documents/${id}`);
}

export function ask(question, documentId) {
  return request("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_id: documentId ?? null }),
  });
}

export function generateQuiz({ startPage, endPage, n, documentId }) {
  if (USE_FIXTURE) return Promise.resolve(fixtures.generate);
  return request("/quiz/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      start_page: startPage,
      end_page: endPage,
      n,
      document_id: documentId ?? null,
    }),
  });
}

export function submitQuiz(quizId, responses) {
  if (USE_FIXTURE) {
    // Score against the fixture's answer key so the view behaves for any
    // selection, rather than always replaying one canned result.
    const marks = fixtures.answerKey.map((correct, position) => ({
      position,
      submitted: responses[position],
      correct_index: correct,
      correct: responses[position] === correct,
      page: fixtures.generate.questions[position].page,
      source_chunk_id: fixtures.chunkIds[position],
      explanation: fixtures.explanations[position],
    }));
    return Promise.resolve({
      quiz_id: quizId,
      score: marks.filter((m) => m.correct).length,
      total: marks.length,
      marks,
    });
  }
  return request("/quiz/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quiz_id: quizId, responses }),
  });
}
