const MAX_GAP_SECONDS = 2.6;
const MAX_DURATION_SECONDS = 24;

function endsSentence(text) {
  return /[.!?…]["'’”)\]]*$/.test(String(text).trim());
}

function speakerKey(segment) {
  return String(segment.speaker ?? segment.speaker_id ?? "");
}

export function buildUtterances(segments) {
  const utterances = [];
  for (const segment of segments) {
    const current = { ...segment };
    const previous = utterances.at(-1);
    if (!previous) {
      utterances.push(current);
      continue;
    }

    const gap = Math.max(0, Number(current.start) - Number(previous.end));
    const combinedDuration = Number(current.end) - Number(previous.start);
    const shouldMerge =
      speakerKey(previous) === speakerKey(current) &&
      !endsSentence(previous.text) &&
      gap <= MAX_GAP_SECONDS &&
      combinedDuration <= MAX_DURATION_SECONDS;

    if (!shouldMerge) {
      utterances.push(current);
      continue;
    }

    previous.end = current.end;
    previous.text =
      `${previous.text?.trim() || ""} ${current.text?.trim() || ""}`.trim();
    if (previous.words || current.words) {
      previous.words = [...(previous.words || []), ...(current.words || [])];
    }
  }
  return utterances;
}
