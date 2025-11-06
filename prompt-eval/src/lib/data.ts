import fs from 'fs';
import path from 'path';
import { DecisionEntry, TrainingDataEntry, OptimizerState } from './types';

// Path to the functions source directory
const FUNCTIONS_SRC = path.resolve(process.cwd(), '..', 'functions', 'src');
const DECISIONS_FILE = path.join(FUNCTIONS_SRC, 'profiles_decisions.jsonl');
const TRAINING_DATA_FILE = path.join(process.cwd(), 'pickup_training_data.jsonl');
const OPTIMIZER_STATE_FILE = path.join(process.cwd(), 'grpo_optimizer_state.json');
const OPENER_PROMPT_FILE = path.join(FUNCTIONS_SRC, 'opener_prompt.yaml');

export function readDecisions(): DecisionEntry[] {
  try {
    if (!fs.existsSync(DECISIONS_FILE)) {
      console.log('Decisions file not found:', DECISIONS_FILE);
      return [];
    }
    const content = fs.readFileSync(DECISIONS_FILE, 'utf8');
    const lines = content.trim().split('\n').filter(line => line.trim());
    return lines.map(line => JSON.parse(line) as DecisionEntry).reverse(); // Most recent first
  } catch (error) {
    console.error('Error reading decisions:', error);
    return [];
  }
}

export function readTrainingData(): TrainingDataEntry[] {
  try {
    if (!fs.existsSync(TRAINING_DATA_FILE)) {
      return [];
    }
    const content = fs.readFileSync(TRAINING_DATA_FILE, 'utf8');
    const lines = content.trim().split('\n').filter(line => line.trim());
    return lines.map(line => JSON.parse(line) as TrainingDataEntry);
  } catch (error) {
    console.error('Error reading training data:', error);
    return [];
  }
}

export function appendTrainingData(entry: TrainingDataEntry): void {
  const line = JSON.stringify(entry) + '\n';
  fs.appendFileSync(TRAINING_DATA_FILE, line, 'utf8');
}

export function getTrainedDecisionIds(): Set<string> {
  const trainingData = readTrainingData();
  return new Set(trainingData.map(t => t.id));
}

export function readOptimizerState(): OptimizerState {
  try {
    if (!fs.existsSync(OPTIMIZER_STATE_FILE)) {
      return {
        experiences: [],
        totalTrainingBatches: 0,
      };
    }
    const content = fs.readFileSync(OPTIMIZER_STATE_FILE, 'utf8');
    return JSON.parse(content) as OptimizerState;
  } catch (error) {
    console.error('Error reading optimizer state:', error);
    return {
      experiences: [],
      totalTrainingBatches: 0,
    };
  }
}

export function writeOptimizerState(state: OptimizerState): void {
  fs.writeFileSync(OPTIMIZER_STATE_FILE, JSON.stringify(state, null, 2), 'utf8');
}

export function readOpenerPrompt(): string {
  try {
    return fs.readFileSync(OPENER_PROMPT_FILE, 'utf8');
  } catch (error) {
    console.error('Error reading opener prompt:', error);
    return '';
  }
}

// Get unrated decisions (not yet in training data)
export function getUnratedDecisions(): DecisionEntry[] {
  const decisions = readDecisions();
  const trainedIds = getTrainedDecisionIds();

  // Create unique ID from decision properties
  const getDecisionId = (d: DecisionEntry) =>
    `${d.userId}_${d.timestamp}`;

  return decisions.filter(d => !trainedIds.has(getDecisionId(d)));
}

// Get recent training data for GRPO batch
export function getRecentTrainingBatch(limit = 20): TrainingDataEntry[] {
  const trainingData = readTrainingData();
  return trainingData.slice(-limit);
}
