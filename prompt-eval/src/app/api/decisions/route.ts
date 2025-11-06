import { NextResponse } from 'next/server';
import { readDecisions, getTrainedDecisionIds } from '@/lib/data';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const page = parseInt(searchParams.get('page') || '1', 10);
  const limit = parseInt(searchParams.get('limit') || '20', 10);
  const filter = searchParams.get('filter') || 'all'; // all, unrated, rated

  const decisions = readDecisions();
  const trainedIds = getTrainedDecisionIds();

  const getDecisionId = (d: { userId: string; timestamp: string }) =>
    `${d.userId}_${d.timestamp}`;

  let filtered = decisions;
  if (filter === 'unrated') {
    filtered = decisions.filter(d => !trainedIds.has(getDecisionId(d)));
  } else if (filter === 'rated') {
    filtered = decisions.filter(d => trainedIds.has(getDecisionId(d)));
  }

  // Add rating status to each decision
  const withRatingStatus = filtered.map(d => ({
    ...d,
    id: getDecisionId(d),
    isRated: trainedIds.has(getDecisionId(d)),
  }));

  const total = withRatingStatus.length;
  const start = (page - 1) * limit;
  const end = start + limit;
  const paginated = withRatingStatus.slice(start, end);

  return NextResponse.json({
    decisions: paginated,
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
    },
  });
}
