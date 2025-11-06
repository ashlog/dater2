import { NextResponse } from 'next/server';
import { appendTrainingData, readTrainingData } from '@/lib/data';
import { DecisionEntry, TrainingDataEntry } from '@/lib/types';

export async function GET() {
  const trainingData = readTrainingData();
  return NextResponse.json({ trainingData });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { decision, rating, regeneratedComment } = body as {
      decision: DecisionEntry & { id: string };
      rating: 'like' | 'dislike';
      regeneratedComment?: string;
    };

    const entry: TrainingDataEntry = {
      id: decision.id,
      timestamp: decision.timestamp,
      originalDecision: decision,
      humanRating: rating,
      humanRatedAt: new Date().toISOString(),
      regeneratedComment,
      regeneratedAt: regeneratedComment ? new Date().toISOString() : undefined,
    };

    appendTrainingData(entry);

    return NextResponse.json({ success: true, entry });
  } catch (error) {
    console.error('Error saving training data:', error);
    return NextResponse.json(
      { error: 'Failed to save training data' },
      { status: 500 }
    );
  }
}
