// Decision entry from profiles_decisions.jsonl
export interface DecisionEntry {
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
  openerPromptHash?: string;
  model?: string;
  deliveryStatus?: 'success' | 'error';
  deliveryError?: { code?: string; message: string };
}

// Training data entry with human feedback
export interface TrainingDataEntry {
  id: string;
  timestamp: string;
  originalDecision: DecisionEntry;
  humanRating: 'like' | 'dislike';
  humanRatedAt: string;
  regeneratedComment?: string;
  regeneratedAt?: string;
}

// GRPO experience from the optimizer
export interface Experience {
  id: string;
  insight: string;
  context: string;
  createdAt: string;
  updatedAt: string;
}

// State for the GRPO optimizer
export interface OptimizerState {
  experiences: Experience[];
  lastTrainedAt?: string;
  totalTrainingBatches: number;
}
