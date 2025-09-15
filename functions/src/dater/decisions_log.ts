import fs from 'fs';
import path from 'path';
import { getQuestionById } from './questions';

export type DecisionEntry = {
  timestamp: string;
  location: { latitude: number; longitude: number };
  userId: string;
  ratingToken: string;
  decision: 'like' | 'skip';
  decisionSource?: 'image' | 'text';
  comment?: string;
  photoUsed?: string | null;
  images: string[];
  prompts: { question: string; answer: string }[];
  profile?: { firstName: string; age?: number };
  imageScores?: { url: string; score: number }[];
  medianImageScore?: number;
};

function resolveDecisionsPath(): string {
  const candidates = [
    path.join(process.cwd(), 'src', 'profiles_decisions.json'),
    path.join(process.cwd(), 'profiles_decisions.json'),
    path.join(__dirname, '..', 'profiles_decisions.json'),
  ];
  for (const p of candidates) {
    try {
      const dir = path.dirname(p);
      if (fs.existsSync(dir)) return p;
    } catch {}
  }
  return path.join(__dirname, 'profiles_decisions.json');
}

const decisionsPath = resolveDecisionsPath();
let decisionsWriteQueue: Promise<void> = Promise.resolve();

export async function appendDecision(entry: DecisionEntry): Promise<void> {
  decisionsWriteQueue = decisionsWriteQueue.then(async () => {
    try {
      let arr: DecisionEntry[] = [];
      try {
        const txt = await fs.promises.readFile(decisionsPath, 'utf8');
        arr = JSON.parse(txt);
        if (!Array.isArray(arr)) arr = [];
      } catch {
        arr = [];
      }
      arr.push(entry);
      await fs.promises.writeFile(decisionsPath, JSON.stringify(arr, null, 2));
    } catch (e) {
      console.error('Failed to append decision', e);
    }
  });
  return decisionsWriteQueue;
}

export function buildPromptsList(filteredAnswers: { questionId: string; response: string }[]): { question: string; answer: string }[] {
  return filteredAnswers.map(a => ({
    question: getQuestionById(a.questionId),
    answer: a.response,
  }));
}
