'use client';

import { useState, useEffect } from 'react';
import { OptimizerState } from '@/lib/types';

interface TrainResult {
  success: boolean;
  optimized?: boolean;
  message?: string;
  operationsApplied?: number;
  newExperienceCount?: number;
  trainingBatchSize?: number;
  skipReason?: string;
  semanticAdvantages?: Array<{ insight: string; context: string }>;
}

export function OptimizerPanel() {
  const [state, setState] = useState<OptimizerState | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);

  const fetchState = async () => {
    try {
      const response = await fetch('/api/optimizer-state');
      const data = await response.json();
      setState(data);
    } catch (error) {
      console.error('Failed to fetch optimizer state:', error);
    }
  };

  useEffect(() => {
    fetchState();
  }, []);

  const handleTrain = async () => {
    setIsTraining(true);
    setTrainResult(null);
    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batchSize: 20 }),
      });
      const result = await response.json();
      setTrainResult(result);
      if (result.success) {
        fetchState(); // Refresh state after training
      }
    } catch (error) {
      console.error('Training failed:', error);
      setTrainResult({ success: false, message: 'Training failed' });
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-4 bg-gradient-to-r from-indigo-500 to-purple-500 text-white flex justify-between items-center"
      >
        <h2 className="text-lg font-bold">GRPO Optimizer</h2>
        <span className="text-2xl">{isExpanded ? '−' : '+'}</span>
      </button>

      {isExpanded && (
        <div className="p-4 space-y-4">
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3">
              <p className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                {state?.experiences.length || 0}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Experiences</p>
            </div>
            <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3">
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {state?.totalTrainingBatches || 0}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Training Batches</p>
            </div>
            <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                {state?.lastTrainedAt
                  ? new Date(state.lastTrainedAt).toLocaleDateString()
                  : 'Never'}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Last Trained</p>
            </div>
          </div>

          {/* Train Button */}
          <button
            onClick={handleTrain}
            disabled={isTraining}
            className="w-full py-3 bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-lg font-semibold transition-all"
          >
            {isTraining ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Training...
              </span>
            ) : (
              'Train with Recent Ratings'
            )}
          </button>

          {/* Train Result */}
          {trainResult && (
            <div
              className={`p-4 rounded-lg ${
                trainResult.success
                  ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200'
                  : 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200'
              }`}
            >
              {trainResult.success ? (
                <div className="space-y-2">
                  <p className="font-semibold">Training Complete!</p>
                  <p>Batch size: {trainResult.trainingBatchSize}</p>
                  <p>Operations applied: {trainResult.operationsApplied}</p>
                  <p>Total experiences: {trainResult.newExperienceCount}</p>
                  {trainResult.semanticAdvantages && trainResult.semanticAdvantages.length > 0 && (
                    <div className="mt-2">
                      <p className="font-medium">New Insights:</p>
                      {trainResult.semanticAdvantages.map((adv, i) => (
                        <p key={i} className="text-sm mt-1">
                          • {adv.insight}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <p>{trainResult.message || trainResult.skipReason || 'Training skipped'}</p>
              )}
            </div>
          )}

          {/* Experiences List */}
          {state?.experiences && state.experiences.length > 0 && (
            <div className="space-y-2">
              <h3 className="font-semibold text-gray-700 dark:text-gray-300">
                Learned Experiences
              </h3>
              <div className="max-h-60 overflow-y-auto space-y-2">
                {state.experiences.map((exp) => (
                  <div
                    key={exp.id}
                    className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-sm"
                  >
                    <p className="font-medium text-indigo-600 dark:text-indigo-400">
                      [{exp.context}]
                    </p>
                    <p className="text-gray-700 dark:text-gray-300 mt-1">{exp.insight}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
