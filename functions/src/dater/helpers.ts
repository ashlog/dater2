export function cleanPickupLineResponse(line: string) {
  const delimiter = '[FINAL THOUGHT]';
  const idx = line.indexOf(delimiter);
  const core = idx >= 0 ? line.substring(idx + delimiter.length) : line;
  return core.trim().replaceAll('"', '').replaceAll('[QUESTION]', '').replaceAll('—', ' - ');
}

export function isBadResponse(comment: string): boolean {
  if (!comment) return true;
  const badPhrases = [
    'pickup line',
    'harmful',
    'inappropriate',
    'respectful',
    'final version',
    'nerate',
    '<EMPTY>',
  ];
  return badPhrases.some(phrase => comment.toLowerCase().includes(phrase.toLowerCase()));
}

