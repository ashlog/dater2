'use client';

import { useState } from 'react';
import { DecisionEntry } from '@/lib/types';

interface ProfileCardProps {
  decision: DecisionEntry & { id: string; isRated: boolean };
  onRate: (id: string, rating: 'like' | 'dislike', regeneratedComment?: string) => void;
  onRegenerate: (decision: DecisionEntry) => Promise<string>;
}

export function ProfileCard({ decision, onRate, onRegenerate }: ProfileCardProps) {
  const [selectedImage, setSelectedImage] = useState(0);
  const [regeneratedComment, setRegeneratedComment] = useState<string | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [showAllPrompts, setShowAllPrompts] = useState(false);

  const handleRegenerate = async () => {
    setIsRegenerating(true);
    try {
      const newComment = await onRegenerate(decision);
      setRegeneratedComment(newComment);
    } catch (error) {
      console.error('Failed to regenerate:', error);
    } finally {
      setIsRegenerating(false);
    }
  };

  const handleRate = (rating: 'like' | 'dislike') => {
    onRate(decision.id, rating, regeneratedComment || undefined);
  };

  const imageScore = decision.imageScores?.find(
    (s) => s.url === decision.images[selectedImage]
  );

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden max-w-2xl mx-auto">
      {/* Header */}
      <div className="p-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-xl font-bold">
              {decision.profile?.firstName || 'Unknown'}, {decision.profile?.age || '?'}
            </h2>
            <p className="text-sm opacity-80">
              {new Date(decision.timestamp).toLocaleString()}
            </p>
          </div>
          <div className="flex gap-2">
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                decision.decision === 'like'
                  ? 'bg-green-400 text-green-900'
                  : 'bg-gray-400 text-gray-900'
              }`}
            >
              Bot: {decision.decision}
            </span>
            {decision.isRated && (
              <span className="px-3 py-1 rounded-full text-sm font-medium bg-blue-400 text-blue-900">
                Rated
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Images */}
      <div className="relative">
        {decision.images.length > 0 ? (
          <>
            <img
              src={decision.images[selectedImage]}
              alt={`Profile photo ${selectedImage + 1}`}
              className="w-full h-80 object-cover"
            />
            {decision.photoUsed === decision.images[selectedImage] && (
              <span className="absolute top-2 left-2 bg-yellow-400 text-yellow-900 px-2 py-1 rounded text-xs font-medium">
                Used for opener
              </span>
            )}
            {imageScore && (
              <span className="absolute top-2 right-2 bg-black/70 text-white px-2 py-1 rounded text-xs">
                Score: {imageScore.score.toFixed(2)}
              </span>
            )}
            {/* Image pagination */}
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1">
              {decision.images.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedImage(i)}
                  className={`w-2 h-2 rounded-full transition-all ${
                    i === selectedImage
                      ? 'bg-white w-4'
                      : 'bg-white/50 hover:bg-white/75'
                  }`}
                />
              ))}
            </div>
          </>
        ) : (
          <div className="w-full h-40 bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
            <span className="text-gray-500">No images</span>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="p-3 bg-gray-50 dark:bg-gray-700 flex justify-between text-sm">
        <span>Median Score: {decision.medianImageScore?.toFixed(2) || 'N/A'}</span>
        <span>Model: {decision.model || 'N/A'}</span>
        <span>Source: {decision.decisionSource || 'N/A'}</span>
      </div>

      {/* Prompts */}
      <div className="p-4 space-y-3">
        <h3 className="font-semibold text-gray-700 dark:text-gray-300">Profile Prompts</h3>
        {decision.prompts.slice(0, showAllPrompts ? undefined : 2).map((prompt, i) => (
          <div key={i} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <p className="text-sm font-medium text-purple-600 dark:text-purple-400">
              {prompt.question}
            </p>
            <p className="text-gray-800 dark:text-gray-200 mt-1">{prompt.answer}</p>
          </div>
        ))}
        {decision.prompts.length > 2 && (
          <button
            onClick={() => setShowAllPrompts(!showAllPrompts)}
            className="text-sm text-purple-600 hover:text-purple-800"
          >
            {showAllPrompts ? 'Show less' : `Show ${decision.prompts.length - 2} more`}
          </button>
        )}
      </div>

      {/* Original Comment */}
      {decision.comment && (
        <div className="px-4 pb-4">
          <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Original Opener
          </h3>
          <div className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900 dark:to-pink-900 rounded-lg p-4">
            <p className="text-gray-800 dark:text-gray-200 italic">
              &ldquo;{decision.comment}&rdquo;
            </p>
          </div>
        </div>
      )}

      {/* Regenerated Comment */}
      {regeneratedComment && (
        <div className="px-4 pb-4">
          <h3 className="font-semibold text-green-600 dark:text-green-400 mb-2">
            Regenerated Opener (with GRPO)
          </h3>
          <div className="bg-gradient-to-r from-green-100 to-teal-100 dark:from-green-900 dark:to-teal-900 rounded-lg p-4">
            <p className="text-gray-800 dark:text-gray-200 italic">
              &ldquo;{regeneratedComment}&rdquo;
            </p>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-600">
        <div className="flex gap-3 justify-center">
          <button
            onClick={() => handleRate('dislike')}
            disabled={decision.isRated}
            className="flex-1 py-3 px-6 bg-red-500 hover:bg-red-600 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-colors"
          >
            Dislike
          </button>
          <button
            onClick={handleRegenerate}
            disabled={isRegenerating}
            className="py-3 px-6 bg-purple-500 hover:bg-purple-600 disabled:bg-purple-400 text-white rounded-lg font-semibold transition-colors"
          >
            {isRegenerating ? 'Generating...' : 'Regenerate'}
          </button>
          <button
            onClick={() => handleRate('like')}
            disabled={decision.isRated}
            className="flex-1 py-3 px-6 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-colors"
          >
            Like
          </button>
        </div>
      </div>

      {/* Delivery Status */}
      {decision.deliveryError && (
        <div className="px-4 pb-4">
          <div className="bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded-lg p-3 text-sm">
            <strong>Error:</strong> {decision.deliveryError.message}
            {decision.deliveryError.code && ` (${decision.deliveryError.code})`}
          </div>
        </div>
      )}
    </div>
  );
}
