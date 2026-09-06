// Renders an answer, replacing citation markers with the page they resolve to.
//
// The page never comes from the model: it emits a passage index, and the page
// is looked up in the `citations` array the API built from chunk metadata. That
// is the same guarantee the backend makes, carried through to the screen.
//
// The pattern accepts full-width CJK brackets as well as ASCII. The model emits
// 【5】 instead of [5] intermittently - in milestone 5 that silently discarded 8
// of 14 citations until the backend regex was widened. Assuming ASCII here
// would reintroduce exactly that bug one layer up.
const MARKER = /[[【](\d+)[\]】]/g;

export default function Answer({ text, citations }) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const parts = [];
  let cursor = 0;

  for (const match of text.matchAll(MARKER)) {
    if (match.index > cursor) parts.push(text.slice(cursor, match.index));

    const citation = byMarker.get(Number(match[1]));
    if (citation) {
      parts.push(
        <span
          key={`${match.index}-${citation.marker}`}
          className="pill"
          title={`chunk #${citation.chunk_id}`}
        >
          p.{citation.page}
        </span>
      );
    } else {
      // Unresolvable marker: shown as written rather than dropped, matching the
      // backend, which records out-of-range markers instead of hiding them.
      parts.push(match[0]);
    }
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) parts.push(text.slice(cursor));
  return <p className="answer">{parts}</p>;
}
