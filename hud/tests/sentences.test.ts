import { describe, expect, it } from "vitest";
import { splitSentences } from "../src/sentences";

describe("splitSentences", () => {
  it("keeps 'Gate No. 2.' in one chunk", () => {
    expect(splitSentences("Gate No. 2.")).toEqual(["Gate No. 2."]);
    expect(splitSentences("RFID card at Gate No. 2.")).toEqual(["RFID card at Gate No. 2."]);
    expect(splitSentences("Meet at Gate No. 2. Then wait.")).toEqual(["Meet at Gate No. 2.", "Then wait."]);
  });
  it("does not split on titles, vs., etc., a.m./p.m., e.g., i.e.", () => {
    for (const t of [
      "Mr. Rao called.", "Mrs. Rao and Dr. Mehta are here.", "St. Mary's school.", "Cats vs. dogs is settled.",
      "Apples, pears, etc. are fine.", "Standup is at 9 a.m. tomorrow.", "Call at 5 p.m. sharp.", "Use a tool, e.g. a hammer.",
      "That is, i.e. the second one.",
    ]) expect(splitSentences(t)).toHaveLength(1);
  });
  it("does not split on a period followed by a digit or a lowercase letter", () => {
    expect(splitSentences("It costs 3. 5 more are due.")).toHaveLength(1);
    expect(splitSentences("Approx. five minutes.")).toHaveLength(1);
    expect(splitSentences("Wait? no, go.")).toHaveLength(1);
  });
  it("still splits real sentences", () => {
    expect(splitSentences("Good evening, Sir. You have two meetings. Shall I read them?")).toEqual([
      "Good evening, Sir.", "You have two meetings.", "Shall I read them?",
    ]);
    expect(splitSentences("Done! Next.")).toEqual(["Done!", "Next."]);
  });
  it("handles empty and single-initial cases", () => {
    expect(splitSentences("")).toEqual([]);
    expect(splitSentences("J. Smith called. Call back.")).toEqual(["J. Smith called.", "Call back."]);
  });
});
