import { NextResponse } from 'next/server';
import { readOptimizerState } from '@/lib/data';

export async function GET() {
  const state = readOptimizerState();
  return NextResponse.json(state);
}
