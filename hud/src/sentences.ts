// Sentence splitting for the HUD display. Mirrors utils/sentences.py (used by TTS chunking): keep the two in step.
//
// A sentence ends at . ! ? followed by whitespace, EXCEPT when:
//   • the word before the period is an abbreviation (No., Mr., Mrs., Dr., St., vs., etc., a.m., p.m., e.g., i.e. …) or a
//     single initial ("J. Smith");
//   • the next word starts with a digit or a lowercase letter (a period before them is almost never a sentence end:
//     "Gate No. 2.", "approx. five", "at 5 p.m. tomorrow").

const ABBREVIATIONS = new Set([
  "no", "nos", "mr", "mrs", "ms", "mx", "dr", "prof", "sr", "jr", "st", "mt", "ft", "vs", "etc", "e.g", "i.e", "a.m", "p.m", "am",
  "pm", "inc", "ltd", "co", "corp", "dept", "approx", "est", "fig", "vol", "rev", "gen", "col", "lt", "sgt", "capt", "gov", "sen",
  "rep", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "cf", "ca", "ave", "blvd", "rd",
]);

const BOUNDARY = /([.!?]+["'”’)\]]*)(\s+)/g;

function isAbbreviation(before: string): boolean {
  const word = before.split(/\s+/).pop() ?? "";
  const w = word.replace(/^["'“‘(\[]+/, "").replace(/\.+$/, "").toLowerCase();
  if (!w) return false;
  if (/^[a-z]$/.test(w)) return true; // an initial
  return ABBREVIATIONS.has(w);
}

export function splitSentences(text: string): string[] {
  const out: string[] = [];
  let start = 0;
  BOUNDARY.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = BOUNDARY.exec(text)) !== null) {
    const end = m.index + m[1].length;
    const next = text.slice(m.index + m[0].length, m.index + m[0].length + 1);
    const endsWithPeriod = m[1].startsWith(".");
    const before = text.slice(start, m.index + 1);
    const lowerOrDigitNext = /[a-z0-9]/.test(next);
    if (endsWithPeriod && m[1] === "." && (isAbbreviation(text.slice(start, m.index)) || lowerOrDigitNext)) continue;
    if (!endsWithPeriod && /[a-z]/.test(next)) continue; // "Wait? no." keeps going
    void before;
    out.push(text.slice(start, end).trim());
    start = m.index + m[0].length;
  }
  const rest = text.slice(start).trim();
  if (rest) out.push(rest);
  return out.filter(Boolean);
}
