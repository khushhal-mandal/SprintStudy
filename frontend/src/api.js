// Fetch wrappers. One error shape, so every view surfaces the same thing.
//
// The API reports upstream failures as 503 with a `detail` naming the cause -
// a stopped container, an exhausted token quota. Swallowing that in favour of a
// generic message would throw away the only useful part of the response.

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

export function listDocuments() {
  return request("/documents");
}

export function ask(question, documentId) {
  return request("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_id: documentId ?? null }),
  });
}

export function generateQuiz({ startPage, endPage, n, documentId }) {
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
  // Scoring is the server's, not this file's. /quiz/generate withholds
  // answer_index precisely so the client cannot mark its own work, and
  // correct_index arrives here only in the /quiz/submit response.
  return request("/quiz/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quiz_id: quizId, responses }),
  });
}
