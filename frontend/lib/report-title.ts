"use client";

const TOPIC_STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "compare",
  "current",
  "evaluate",
  "explain",
  "find",
  "for",
  "from",
  "how",
  "in",
  "map",
  "of",
  "on",
  "or",
  "recent",
  "research",
  "state",
  "survey",
  "the",
  "to",
  "what",
  "with",
]);

function cleanPrompt(value: string): string {
  return value
    .replace(/^as of [A-Za-z]+ \d{1,2}, \d{4},?\s*/i, "")
    .replace(/\b(evidence requirements|avoid seo|separate proven).*$/i, "")
    .replace(/^(research|evaluate|survey|compare|find|map|explain)\s+/i, "")
    .trim();
}

function promptKeywords(question: string, maxWords: number): string[] {
  const cleaned = cleanPrompt(question);
  const words = cleaned.match(/[A-Za-z0-9][A-Za-z0-9/-]*/g) ?? [];
  const selected = words.filter(
    (word) => !TOPIC_STOPWORDS.has(word.toLowerCase()) && !/^\d{4}$/.test(word),
  );
  return (selected.length >= 3 ? selected : words).slice(0, maxWords);
}

function titleWord(word: string): string {
  return word
    .split("-")
    .map((part) => {
      if (part.toUpperCase() === part) return part;
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join("-");
}

export function deriveConversationTopic(question: string): string {
  const topic = promptKeywords(question, 6).join(" ");
  return topic || "Research Topic";
}

export function deriveReportTitle(question: string): string {
  const title = promptKeywords(question, 12).map(titleWord).join(" ");
  return title || "Grounded Research Report";
}

export function isGenericReportTitle(value: string | null | undefined): boolean {
  const normalized = (value ?? "").trim().toLowerCase();
  return (
    normalized === "" ||
    normalized === "research report" ||
    normalized === "final report" ||
    normalized === "grounded research report"
  );
}
