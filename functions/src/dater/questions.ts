import { qna } from './qna';

const questionMap = new Map<string, string>();
qna.prompts.forEach(prompt => {
  questionMap.set(prompt.id, prompt.prompt);
});

export function getQuestionById(id: string): string {
  return questionMap.get(id) || id;
}

