import { NextResponse } from 'next/server';
import { regenerateOpener } from '@/lib/opener-generator';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { profilePrompt, herName, imageUrl, caption, useExperiences } = body;

    if (!profilePrompt || !herName) {
      return NextResponse.json(
        { error: 'profilePrompt and herName are required' },
        { status: 400 }
      );
    }

    const generatedComment = await regenerateOpener({
      profilePrompt,
      herName,
      imageUrl,
      caption,
      useExperiences: useExperiences !== false,
    });

    return NextResponse.json({ generatedComment });
  } catch (error) {
    console.error('Error regenerating opener:', error);
    return NextResponse.json(
      { error: 'Failed to regenerate opener' },
      { status: 500 }
    );
  }
}
