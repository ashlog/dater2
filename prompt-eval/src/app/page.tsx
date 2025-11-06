'use client';

import { useState, useEffect, useCallback } from 'react';
import { ProfileCard } from '@/components/ProfileCard';
import { OptimizerPanel } from '@/components/OptimizerPanel';
import { DecisionEntry } from '@/lib/types';

interface DecisionWithMeta extends DecisionEntry {
  id: string;
  isRated: boolean;
}

interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

type FilterType = 'all' | 'unrated' | 'rated';

export default function Home() {
  const [decisions, setDecisions] = useState<DecisionWithMeta[]>([]);
  const [pagination, setPagination] = useState<PaginationInfo | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [filter, setFilter] = useState<FilterType>('unrated');
  const [isLoading, setIsLoading] = useState(true);
  const [ratedCount, setRatedCount] = useState(0);

  const fetchDecisions = useCallback(async (page: number, filterType: FilterType) => {
    setIsLoading(true);
    try {
      const response = await fetch(
        `/api/decisions?page=${page}&limit=5&filter=${filterType}`
      );
      const data = await response.json();
      setDecisions(data.decisions);
      setPagination(data.pagination);
    } catch (error) {
      console.error('Failed to fetch decisions:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDecisions(currentPage, filter);
  }, [currentPage, filter, fetchDecisions]);

  const handleRate = async (
    id: string,
    rating: 'like' | 'dislike',
    regeneratedComment?: string
  ) => {
    const decision = decisions.find((d) => d.id === id);
    if (!decision) return;

    try {
      await fetch('/api/training-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, rating, regeneratedComment }),
      });

      // Update local state
      setDecisions((prev) =>
        prev.map((d) => (d.id === id ? { ...d, isRated: true } : d))
      );
      setRatedCount((prev) => prev + 1);

      // Refresh if filtering by unrated
      if (filter === 'unrated') {
        fetchDecisions(currentPage, filter);
      }
    } catch (error) {
      console.error('Failed to save rating:', error);
    }
  };

  const handleRegenerate = async (decision: DecisionEntry): Promise<string> => {
    const profilePrompt = decision.prompts
      .map((p) => `${p.question}: ${p.answer}`)
      .join('\n');

    const response = await fetch('/api/regenerate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profilePrompt,
        herName: decision.profile?.firstName || 'Unknown',
        imageUrl: decision.decisionSource === 'image' ? decision.photoUsed : undefined,
        caption: decision.decisionSource === 'image' ? '' : undefined,
        useExperiences: true,
      }),
    });

    const data = await response.json();
    return data.generatedComment;
  };

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-gradient-to-r from-purple-600 to-pink-600 text-white py-6 shadow-lg">
        <div className="max-w-6xl mx-auto px-4">
          <h1 className="text-3xl font-bold">Prompt Evaluation UI</h1>
          <p className="text-purple-200 mt-1">
            Rate pickup lines to train the GRPO optimizer
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Optimizer Panel */}
          <div className="lg:col-span-1 space-y-4">
            <OptimizerPanel />

            {/* Stats */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4">
              <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">
                Session Stats
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-green-600">{ratedCount}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Rated This Session
                  </p>
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-purple-600">
                    {pagination?.total || 0}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Total {filter}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Decisions */}
          <div className="lg:col-span-2 space-y-4">
            {/* Filter Controls */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4">
              <div className="flex flex-wrap gap-2">
                {(['unrated', 'rated', 'all'] as FilterType[]).map((f) => (
                  <button
                    key={f}
                    onClick={() => {
                      setFilter(f);
                      setCurrentPage(1);
                    }}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      filter === f
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Decisions List */}
            {isLoading ? (
              <div className="flex justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-purple-500 border-t-transparent"></div>
              </div>
            ) : decisions.length === 0 ? (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 text-center">
                <p className="text-gray-600 dark:text-gray-400 text-lg">
                  No {filter} decisions found
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {decisions.map((decision) => (
                  <ProfileCard
                    key={decision.id}
                    decision={decision}
                    onRate={handleRate}
                    onRegenerate={handleRegenerate}
                  />
                ))}
              </div>
            )}

            {/* Pagination */}
            {pagination && pagination.totalPages > 1 && (
              <div className="flex justify-center gap-2 py-4">
                <button
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-4 py-2 bg-white dark:bg-gray-800 rounded-lg shadow disabled:opacity-50"
                >
                  Previous
                </button>
                <span className="px-4 py-2 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 rounded-lg">
                  {currentPage} / {pagination.totalPages}
                </span>
                <button
                  onClick={() =>
                    setCurrentPage((p) => Math.min(pagination.totalPages, p + 1))
                  }
                  disabled={currentPage === pagination.totalPages}
                  className="px-4 py-2 bg-white dark:bg-gray-800 rounded-lg shadow disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
