#!/usr/bin/env npx ts-node
/**
 * Opener Quality Eval System with LLM Judge Personas
 *
 * Two modes:
 *   Historical: Score existing openers from profiles_decisions.jsonl
 *   Generate:   Regenerate fresh openers with current prompt across models, then judge
 *
 * Usage:
 *   npx ts-node src/eval_openers.ts --samples 30
 *   npx ts-node src/eval_openers.ts --samples 30 --hash 57cd188e
 *   npx ts-node src/eval_openers.ts --generate --samples 10 --models anthropic/claude-sonnet-4.5,anthropic/claude-haiku-4.5
 */

import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import {
  generateOpenersFromYamlBatch,
  generateOpenersFromYamlBatchBestOfN,
  getOpenerPromptMetadata,
  setLLMProvider,
  cleanupLLMProvider,
  getCostSummary,
  resetCostTracker,
  setOpenerPromptOverride,
  setLLMThinkingConfig,
  chat as llmChat,
  ChatCompletionMessageParam,
} from './dater/llm';

dotenv.config();

// --- Types ---

interface Decision {
  timestamp: string;
  decision: string;
  decisionSource: string;
  comment?: string;
  prompts: { question: string; answer: string }[];
  profile: { firstName: string; age: number };
  model: string;
  openerPromptHash?: string;
  userId?: string;
}

interface DimensionScore {
  score: number;
  rationale: string;
}

interface JudgeScores {
  reply_likelihood: DimensionScore;
  humor_quality: DimensionScore;
  creativity: DimensionScore;
  specificity: DimensionScore;
  human_likeness: DimensionScore;
  closer_strength: DimensionScore;
}

interface EvalResult {
  model: string;
  comment: string;
  promptQuestion: string;
  promptAnswer: string;
  profileName: string;
  profileAge: number;
  decisionSource: string;
  scores: JudgeScores;
}

interface ModelAggregate {
  model: string;
  samples: number;
  reply_likelihood: number;
  humor_quality: number;
  creativity: number;
  specificity: number;
  human_likeness: number;
  closer_strength: number;
  avg: number;
}

type EvalArgs =
  | { mode: 'historical'; samples: number; hash?: string }
  | { mode: 'generate'; samples: number; models: string[]; prompts?: string[]; bestOf?: number };

// --- Config ---

const JUDGE_MODEL = 'claude-opus-4-6';
const CONCURRENCY = 6;
const GEN_CONCURRENCY = 6;


// --- System prompt (cached across all judge calls) ---
// Must be 4096+ tokens for Haiku 4.5 prompt caching to activate.
// Detailed rubrics and calibration examples also improve eval consistency.

const JUDGE_SYSTEM_PROMPT = `You are an expert evaluator of dating app opener messages on Hinge. You have deep experience with what makes openers effective on dating apps: what gets replies, what feels authentic, and what screams "AI-generated." You will evaluate each opener from 5 distinct perspectives, providing a score (1-10) and a concise one-line rationale for each.

## Context: What is Hinge?

Hinge is a dating app where users create profiles with photos and text prompts (e.g., "A life goal of mine" → "Learn to surf"). When someone wants to express interest, they send a "like" along with an optional comment (the "opener") on a specific prompt or photo. The opener is the user's first message and serves as a conversation starter. Good openers feel personal, show genuine interest in the specific person, and make it easy to reply.

## Dimension 1: REPLY_LIKELIHOOD (1-10)

Would this opener actually get a reply? Score based on practical effectiveness as a conversation starter.

### Scoring Rubric:
- **9-10**: Highly likely to get a reply. Shows specific, genuine engagement with their prompt. Asks a natural question or makes a comment that demands a response. Right length (not too short, not overwhelming). Creates conversational momentum.
- **7-8**: Good chance of reply. Shows engagement with their content, has a hook or question, feels personal. Minor issues like being slightly generic or a bit too clever.
- **5-6**: Average. Shows some engagement but could be more specific. May be too generic, too short to stand out, or missing a clear invitation to respond. Not bad, but forgettable.
- **3-4**: Below average. Feels like it could be sent to anyone. Too generic, too long, too intense, or too clever. Doesn't create conversational flow. The recipient might read it and not know what to say back.
- **1-2**: Very unlikely to get a reply. Creepy, off-putting, completely generic ("hey"), way too long, or so convoluted that it's confusing. Dead-end comment with no invitation to respond.

### Key factors:
- Does it reference something SPECIFIC from their prompt/answer/photo?
- Does it contain a question or hook that makes replying easy and natural?
- Is the length appropriate? (1-2 sentences is ideal; 3+ sentences can be overwhelming)
- Does it feel like a conversation starter, not a monologue?
- Would it stand out among the 10+ likes they receive daily?
- Does it create a sense of shared interest or connection?

## Dimension 2: HUMOR_QUALITY (1-10)

Is the humor natural and effective, or forced and tryhard? Does it actually make someone laugh or smile?

### Scoring Rubric:
- **9-10**: Genuinely funny. The humor feels effortless, like something a naturally witty person would say in conversation. Makes you actually smile or laugh. The comedic timing and word choice feel natural, not workshopped.
- **7-8**: Solid humor. Clearly intentional wit that mostly lands. Feels like something a clever person might say, though maybe slightly polished. Good observation or callback to their content.
- **5-6**: Mild humor. There's an attempt at being funny that somewhat works. Not cringe-worthy, but not memorable either. The joke or wordplay is serviceable but unremarkable.
- **3-4**: Forced humor. The joke or wordplay feels manufactured. Puns that don't land, observations that aren't interesting, or callbacks that feel mechanical.
- **1-2**: Cringe. Bad puns, forced references, jokes that don't make sense, or attempts at being funny that come across as awkward.

### Key factors:
- Does the humor flow naturally from the content, or does it feel bolted on?
- Would a naturally funny person say this in real conversation?
- Is it observational humor (good) or constructed wordplay (often bad)?
- Does the humor serve the conversation or just show off cleverness?
- Would this make someone laugh, smile, or cringe?

## Dimension 3: CREATIVITY (1-10)

How clever, unexpected, or original is the angle? Does it find a surprising way to engage with the content?

### Scoring Rubric:
- **9-10**: Brilliant angle. Finds a genuinely surprising or clever way to engage with the prompt that most people would never think of. Uses a strong comedic device (reversal, literal misread, absurd escalation, great pun). The recipient would think "wow, that's clever."
- **7-8**: Clearly creative. Uses an interesting angle or device — a solid pun, a funny reframe, or an unexpected observation. Shows real thought and effort, not just the obvious response.
- **5-6**: Some creativity. There's a mild attempt at an angle, but it's fairly predictable. The response is what most clever people might come up with. Not boring, but not surprising either.
- **3-4**: Generic approach. Basically just responds to the content directly with no creative twist. Asks the obvious follow-up question. No comedic device, no interesting angle, just... engagement.
- **1-2**: Zero creativity. Could be sent to anyone about anything. No angle, no device, no twist. Pure filler or the most obvious possible response.

### Comedic devices that score high (when executed well):
- **Puns/wordplay**: Double meanings, homonyms, sound-alikes. The gold standard when they land naturally. E.g., "significant otter", "cayenne-destined", "four fox sake"
- **Literal misread**: Taking something absurdly literally. E.g., "dog person" → "I'm full human, should I still comment?"
- **Reversal**: Flipping expectations. E.g., "I hate competition" → "I bet I'm less competitive than you"
- **Absurd escalation**: Over-the-top scenario building. E.g., "clumsiest alive" → "specify alive because the dead ones are certainly clumsier"
- **Faux-serious**: Mock-formal tone on silly topic
- **Deadpan reframe**: Treating something mundane as suspicious or consequential

### What scores LOW on creativity:
- The "rephrase + question" format (no device, just restating and asking)
- Saying "that's amazing/cool/elite" + generic follow-up
- Asking the obvious question anyone would ask
- Profile-summarizing without any twist

## Dimension 4: SPECIFICITY (1-10)

Does the opener engage with something concrete and specific from this person's profile, or could it be sent to anyone?

### Scoring Rubric:
- **9-10**: Laser-focused on ONE specific detail from their prompt or answer. The opener couldn't work for any other profile. It picks a single hook and commits fully to it.
- **7-8**: Clearly references their specific content, but might be slightly generic in how it engages with it. Shows they read the profile carefully.
- **5-6**: References the topic area but not a specific detail. Could work for anyone with a similar prompt topic. E.g., responding to any cooking prompt with a generic cooking question.
- **3-4**: Very loosely connected to their profile. The opener touches on the general theme but doesn't engage with anything specific they said. Could be sent to many profiles.
- **1-2**: Completely generic. Could be sent to literally anyone regardless of their profile content. No connection to this specific person.

### Key factors:
- Does it pick ONE hook and commit, or try to address multiple things?
- Does it reference a SPECIFIC detail (a car model, a place name, a number) from their answer?
- Would this opener make sense if sent to a different profile with a different prompt? If yes, score lower.
- Single-hook commitment is a strong signal: picking the strongest detail and ignoring everything else shows good judgment.

## Dimension 5: HUMAN_LIKENESS (1-10)

How likely is this written by a real human vs. an AI? This is a detection task.

### Scoring Rubric:
- **9-10**: Definitely human. Natural voice, casual style, maybe a typo or informal grammar. Feels like a real person typing on their phone. Has personality and voice that feels authentic and individual.
- **7-8**: Probably human. Mostly natural-sounding but slightly too polished or well-constructed for a casual dating app message. Could go either way but leans human.
- **5-6**: Ambiguous. Some natural elements mixed with some AI-like qualities. Hard to tell either way. May have one or two red flags but nothing conclusive.
- **3-4**: Probably AI. Multiple red flags present. Formulaic structure, overly polished language, or patterns that are characteristic of AI-generated text. Feels like it was optimized rather than naturally written.
- **1-2**: Obviously AI. Screams AI-generated. Multiple strong red flags, formulaic structure, mechanical engagement pattern, and unnatural language choices that no real person would make in a dating app context.

### RED FLAGS that indicate AI-generated text (each flag should lower the score):
1. **Formulaic structure**: The opener follows a predictable pattern like "rephrases their prompt/answer → asks a witty follow-up question." Real humans don't systematically rephrase before asking questions.
2. **Filler transitions**: Phrases like "ok hold up", "wait", "ok so", "I need to know" used as openers. These are attempts to simulate spontaneity that real humans rarely use as their FIRST message to a stranger.
3. **Forced wordplay**: Puns or clever constructions that feel manufactured rather than spontaneous. Especially wordplay that references multiple parts of their profile in a single message.
4. **Over-enthusiasm**: Excessive energy or excitement that feels performative. Too many exclamation marks, all-caps words, or breathless reactions to mundane content.
5. **Too-perfect structure**: Dating app messages from real humans are usually messy — incomplete sentences, lowercase, casual punctuation. AI messages tend to be grammatically perfect with proper punctuation.
6. **Emoji mirroring**: Copying emoji usage from the profile or using emojis in a pattern that matches the profile's energy too precisely.
7. **Unnatural specificity**: Pulling in multiple specific details from the profile into a single message in a way that feels systematic rather than organic.
8. **Balanced construction**: Messages where the humor, question, and personal touch are too perfectly balanced, as if optimized by an algorithm.
9. **Characteristic AI phrases**: Certain turns of phrase that are hallmarks of AI text — "this is my kind of chaos", "I need to hear this story", "tell me everything", "this determines everything", "I'm keeping this..."
10. **Lack of personality**: The message could have been written by anyone — it doesn't reveal anything about the sender's actual personality, interests, or voice. It's purely reactive to the recipient's content.
11. **Dashes and em-dashes**: Overuse of dashes (—) or hyphens (-) to create dramatic pauses or connect thoughts. Real dating app messages rarely use em-dashes.
12. **Overly smooth transitions**: Real people tend to just blurt things out. AI tends to create smooth, logical connections between ideas.

### INDICATORS of genuine human writing:
- Typos, autocorrect errors, or casual misspellings
- Incomplete sentences or fragments
- All lowercase or inconsistent capitalization
- Messages that reveal something about the sender, not just react to the recipient
- Genuine awkwardness or vulnerability
- References to personal experiences that aren't directly prompted by the profile
- Simple, direct questions without clever framing
- Natural casual tone like texting a friend

## Dimension 6: CLOSER_STRENGTH (1-10)

How well does the opener END? The last 3-4 words are what she remembers. Score based on whether the closer lands with impact or trails off.

### Scoring Rubric:
- **9-10**: The last words ARE the punch. The closer is the strongest part of the opener — a punchline, a mic-drop, or a short question that demands a reply. Reading only the ending gives you the full comedic/emotional hit.
- **7-8**: Strong ending that works but could be slightly tighter. The closer carries momentum but isn't the absolute peak of the message.
- **5-6**: Forgettable ending. The best part of the opener is in the middle, not at the end. The closer is functional but doesn't land with impact. Slight trailing off.
- **3-4**: The closer deflates the opener. Hedges ("though", "but still", "just saying"), self-deprecating retreats that kill confidence, or explanations that undercut the punchline. The opener would be BETTER if the last few words were cut.
- **1-2**: The closer actively kills the opener. Apologetic tone, permission-seeking ("if that's cool"), or so weak it makes her forget the good parts that came before.

### Key factors:
- Read ONLY the last 3-4 words. Do they land with confidence or trail off?
- Could you cut the last few words and IMPROVE the opener? If yes, score 5 or below.
- Does the ending hedge or retreat from the joke's confidence? ("trainable though", "no promises", "i'll try")
- Does it end on something she can GRAB ONTO to reply?
- Is the punchline at the END or buried in the middle?

### Strong closer patterns:
- Short question that demands an answer: "what happened", "should i be concerned"
- Confident mic-drop: "no refunds", "this matters"
- Self-deprecating that still OWNS it: "wrong department though" (owns the joke, doesn't retreat)

### Weak closer patterns:
- Hedging retreats: "trainable though", "but i'll try", "no pressure"
- Permission-seeking: "if that's cool", "if you're into that"
- Explaining the joke: "because X", "which means Y"
- Trailing qualifiers: "just saying", "or something", "idk"

## Output Format

You must respond with ONLY valid JSON, no other text before or after. Use this exact structure:
{"reply_likelihood": {"score": N, "rationale": "..."}, "humor_quality": {"score": N, "rationale": "..."}, "creativity": {"score": N, "rationale": "..."}, "specificity": {"score": N, "rationale": "..."}, "human_likeness": {"score": N, "rationale": "..."}, "closer_strength": {"score": N, "rationale": "..."}}

Where N is an integer from 1 to 10 and rationale is a single concise sentence explaining the score.

## Calibration Examples

These examples show how to score openers of different quality levels. Study these carefully to calibrate your scoring.

### Example 1 (High scores across the board):
Profile: Sarah, 26
Prompt: "I know the best spot in town for" → "finding random cats to pet"
Opener: "what neighborhood we talking? i have a mental map of cat hotspots"
Expected: {"reply_likelihood": {"score": 8, "rationale": "Specific engagement, easy question to answer, shows shared interest"}, "humor_quality": {"score": 7, "rationale": "Naturally funny concept of a mental cat map without forcing the joke"}, "creativity": {"score": 7, "rationale": "The 'mental map of cat hotspots' is an unexpected personal detail that reframes the conversation"}, "specificity": {"score": 8, "rationale": "Engages directly with the cat-petting detail and adds a specific personal angle"}, "human_likeness": {"score": 9, "rationale": "All lowercase, casual tone, reveals personal quirk, feels like a real text"}, "closer_strength": {"score": 8, "rationale": "'cat hotspots' is a fun image to end on — memorable and easy to riff on"}}

### Example 2 (Generic AI rephrase):
Profile: Emily, 24
Prompt: "A life goal of mine" → "Learn to make pasta from scratch"
Opener: "ok homemade pasta is elite - are we talking ravioli level ambition or starting with the basics?"
Expected: {"reply_likelihood": {"score": 7, "rationale": "Good engagement and question, though slightly structured"}, "humor_quality": {"score": 5, "rationale": "Mild humor with 'elite' and 'ambition' framing, nothing actually funny"}, "creativity": {"score": 3, "rationale": "No creative device — just restates 'pasta' and asks the obvious follow-up about what kind"}, "specificity": {"score": 6, "rationale": "References pasta specifically but the either/or question is generic enough for any cooking prompt"}, "human_likeness": {"score": 4, "rationale": "'ok' opener + dash + either/or question is a common AI formula"}, "closer_strength": {"score": 5, "rationale": "Ends on 'starting with the basics' — functional but forgettable, the either/or fizzles"}}

### Example 3 (Formulaic AI pattern):
Profile: Jessica, 27
Prompt: "You'd never know it, but I" → "can solve a Rubik's cube in under 2 minutes"
Opener: "ok hold up - a Rubik's cube in under 2 minutes?? I need the full origin story. Are we talking casual hobby or competitive speedcuber?"
Expected: {"reply_likelihood": {"score": 6, "rationale": "Has a question but feels like an interview rather than banter"}, "humor_quality": {"score": 3, "rationale": "No actual humor, just performed surprise and a formulaic question"}, "creativity": {"score": 2, "rationale": "Zero creative device — just rephrases what she said and asks for elaboration"}, "specificity": {"score": 7, "rationale": "Does reference the specific 2-minute Rubik's cube detail"}, "human_likeness": {"score": 2, "rationale": "Classic AI formula: 'ok hold up' + rephrase + demand for story + either/or question"}, "closer_strength": {"score": 4, "rationale": "Ends on 'competitive speedcuber?' — the either/or trails off instead of landing"}}

### Example 4 (Simple and effective):
Profile: Rachel, 25
Prompt: "My simple pleasures" → "a perfect sunset and a glass of wine"
Opener: "rooftop or beach sunset? important distinction"
Expected: {"reply_likelihood": {"score": 8, "rationale": "Simple, playful question that's easy and fun to answer"}, "humor_quality": {"score": 6, "rationale": "Light humor in treating it as a serious question, charming not hilarious"}, "creativity": {"score": 5, "rationale": "Decent reframe as a serious distinction, but a fairly obvious follow-up question"}, "specificity": {"score": 7, "rationale": "Picks the sunset detail specifically and adds a new dimension to it"}, "human_likeness": {"score": 8, "rationale": "Short, direct, no unnecessary framing or filler — feels like a real person"}, "closer_strength": {"score": 9, "rationale": "'important distinction' is a perfect deadpan mic-drop — two words that carry gravity and invite a reply"}}

### Example 5 (Great pun — creative but AI-sounding):
Profile: Amanda, 25
Prompt: "A random fact I love" → "otters hold paws while they sleep so they don't drift apart"
Opener: "can I be your significant otter?"
Expected: {"reply_likelihood": {"score": 8, "rationale": "Charming, memorable, easy to respond to with a laugh or playful yes"}, "humor_quality": {"score": 8, "rationale": "The pun genuinely lands — it's clever without being forced and fits the context perfectly"}, "creativity": {"score": 9, "rationale": "'Significant otter' is a brilliant pun that transforms the otter fact into a flirty proposition"}, "specificity": {"score": 9, "rationale": "Only works for this exact prompt — the pun is built entirely from her specific fact"}, "human_likeness": {"score": 7, "rationale": "Lowercase, short, natural — but the pun is almost too perfect for a casual first message"}, "closer_strength": {"score": 10, "rationale": "'significant otter' IS the punchline — the entire opener builds to these two words and they land perfectly"}}

### Example 6 (Deadpan genius):
Profile: Sam, 26
Prompt: "You'd never know it, but I" → "have been to 23 countries"
Opener: "23 is such a specific number to stop at. what happened"
Expected: {"reply_likelihood": {"score": 8, "rationale": "Funny angle that reframes their humble brag as suspicious, invites storytelling"}, "humor_quality": {"score": 8, "rationale": "Deadpan 'what happened' is genuinely funny — treats travel as if something went wrong"}, "creativity": {"score": 9, "rationale": "Brilliant reframe — instead of being impressed, acts like 23 is suspicious. Nobody expects that angle."}, "specificity": {"score": 9, "rationale": "Focuses on the specific number '23' — couldn't work for any other travel count"}, "human_likeness": {"score": 9, "rationale": "No capitals, dry humor, period instead of question mark — very human texting style"}, "closer_strength": {"score": 10, "rationale": "'what happened' — two words, maximum impact, demands an answer, IS the punchline"}}

### Example 7 (Literal misread device):
Profile: Kate, 28
Prompt: "You should leave a comment if" → "you're a dog person"
Opener: "I'm actually full human. Should I still leave a comment?"
Expected: {"reply_likelihood": {"score": 7, "rationale": "Absurd enough to stand out, invites a playful response"}, "humor_quality": {"score": 8, "rationale": "Literal misread of 'dog person' as species is genuinely funny and unexpected"}, "creativity": {"score": 9, "rationale": "Takes 'dog person' literally as a human-dog hybrid — classic literal misread device perfectly executed"}, "specificity": {"score": 8, "rationale": "Built entirely around her specific 'dog person' phrase"}, "human_likeness": {"score": 7, "rationale": "Proper punctuation and structure are slightly formal but the humor style feels human"}, "closer_strength": {"score": 8, "rationale": "'should I still comment?' ends on a question that invites her to play along with the bit"}}

### Example 8 (Trying too hard — low creativity despite effort):
Profile: Taylor, 24
Prompt: "I get myself out of a funk by" → "baking sourdough and blasting 90s hip hop"
Opener: "sourdough AND 90s hip hop?? this is the most chaotic self-care I've ever heard and I'm absolutely here for it - what's the go-to track, biggie or tupac?"
Expected: {"reply_likelihood": {"score": 5, "rationale": "Too long and energetic for a first message, multiple questions compete for attention"}, "humor_quality": {"score": 4, "rationale": "'Chaotic self-care' is a social media phrase, not natural humor"}, "creativity": {"score": 3, "rationale": "No creative device — just reacts with performed surprise and asks the obvious music question"}, "specificity": {"score": 4, "rationale": "References both sourdough AND hip hop — profile-summarizing instead of single-hook commitment"}, "human_likeness": {"score": 2, "rationale": "Double question marks, 'absolutely here for it', combining multiple profile references — all AI hallmarks"}, "closer_strength": {"score": 4, "rationale": "'biggie or tupac?' is the obvious question anyone would ask — the closer defaults to a generic either/or"}}

### Example 9 (Direct quote — perfectly human, moderate creativity):
Profile: Lisa, 23
Prompt: "I'll fall for you if" → "you can quote The Office"
Opener: "bears beets battlestar galactica"
Expected: {"reply_likelihood": {"score": 7, "rationale": "Perfect reference that shows shared interest, easy to riff on"}, "humor_quality": {"score": 7, "rationale": "Using the quote directly is funnier than explaining it"}, "creativity": {"score": 6, "rationale": "It's the most famous Office quote — obvious choice, but letting the quote speak for itself is smart"}, "specificity": {"score": 9, "rationale": "Only works because she specifically asked for Office quotes"}, "human_likeness": {"score": 10, "rationale": "No framing, no question, just the reference itself — exactly what a real fan would do"}, "closer_strength": {"score": 9, "rationale": "'battlestar galactica' — the quote builds in absurdity and ends on the most ridiculous item"}}

### Example 10 (Reversal device):
Profile: Jess, 25
Prompt: "I'm overly competitive about" → "absolutely nothing - I hate competition"
Opener: "I bet I'm less competitive than you"
Expected: {"reply_likelihood": {"score": 8, "rationale": "Creates instant playful tension that begs a response"}, "humor_quality": {"score": 9, "rationale": "The irony of competitively claiming to be non-competitive is effortlessly funny"}, "creativity": {"score": 10, "rationale": "Perfect reversal — turns her anti-competition stance into a competition. One of the best possible angles."}, "specificity": {"score": 9, "rationale": "Only works for this exact prompt about hating competition"}, "human_likeness": {"score": 9, "rationale": "Short, punchy, no filler — feels like a genuinely witty person's instinct"}, "closer_strength": {"score": 10, "rationale": "'than you' — the entire joke hinges on these two words landing at the end, and they do perfectly"}}

## Common AI Patterns to Watch For

These patterns appear frequently in AI-generated openers and should lower the HUMAN_LIKENESS score:

1. **The "ok [reaction]" opener**: Starting with "ok" followed by an exaggerated reaction. E.g., "ok hold up", "ok so", "ok wait", "ok this is..."
2. **The rephrase-and-question**: Restating what the person said in slightly different words, then asking a follow-up. This is the single most common AI opener pattern.
3. **Performed spontaneity**: Fake surprise or excitement that feels calculated. "WAIT", "I NEED to know", "stop everything", "ok but seriously"
4. **The either/or question**: "are we talking X or Y?" — especially when both options are things the person might enjoy.
5. **Profile-combining**: Referencing multiple distinct parts of a profile in one message, as if systematically scanning all available content.
6. **Superlative reactions**: "this is the best thing", "most chaotic", "absolute elite taste" — inflated reactions to ordinary profile content.
7. **The em-dash connector**: Using em-dashes or hyphens to create dramatic pauses or connect thoughts in ways that read as written rather than typed.
8. **Bracket humor**: Enclosing asides in parentheses or brackets that feel like stage directions rather than natural thought.

## Important Notes

- Be critical. Most AI-generated openers cluster around 3-5 on human_likeness and 3-5 on creativity. It's rare for an AI opener to score 8+ on either.
- All 6 dimensions are independent. A boring but genuine message might score 7 on reply likelihood but 3 on humor and creativity. A brilliantly creative opener might still sound obviously AI.
- Don't let one strong dimension inflate others. Score each independently.
- The rationale should be specific about what made you give that score, not generic praise or criticism.
- When scoring HUMAN_LIKENESS, count the number of red flags present. 0 flags = 8-10, 1 flag = 6-7, 2 flags = 4-5, 3+ flags = 1-3.
- When scoring CREATIVITY, ask: "What comedic device is being used?" If you can't name one (pun, reversal, literal misread, absurd escalation, faux-serious, deadpan reframe), the creativity score should be 4 or below.
- When scoring SPECIFICITY, ask: "Would this opener work on a different profile with a similar topic?" If yes, score 5 or below.
- When scoring CLOSER_STRENGTH, read ONLY the last 3-4 words in isolation. Ask: "Do these words land with impact? Would cutting them improve the opener?" If cutting them helps, score 5 or below.`;

// --- Helpers ---

function parseArgs(): EvalArgs {
  const args = process.argv.slice(2);
  let samples = 30;
  let generate = false;
  let hash: string | undefined;
  let models: string[] = ['claude-opus-4-6'];
  let prompts: string[] | undefined;
  let thinkingBudget: number | undefined;
  let bestOf: number | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--samples' && args[i + 1]) {
      samples = parseInt(args[i + 1], 10);
    } else if (args[i] === '--generate') {
      generate = true;
    } else if (args[i] === '--hash' && args[i + 1]) {
      hash = args[i + 1];
    } else if (args[i] === '--models' && args[i + 1]) {
      models = args[i + 1].split(',').map((m) => m.trim());
    } else if (args[i] === '--prompts' && args[i + 1]) {
      prompts = args[i + 1].split(',').map((p) => p.trim());
    } else if (args[i] === '--thinking' && args[i + 1]) {
      thinkingBudget = parseInt(args[i + 1], 10);
    } else if (args[i] === '--best-of' && args[i + 1]) {
      bestOf = parseInt(args[i + 1], 10);
    }
  }

  // Enable extended thinking if requested
  if (thinkingBudget && thinkingBudget > 0) {
    setLLMThinkingConfig({ type: 'enabled', budget_tokens: thinkingBudget });
    console.log(`  Extended thinking enabled: budget_tokens=${thinkingBudget}`);
  }

  if (generate) {
    return { mode: 'generate', samples, models, prompts, bestOf };
  }
  return { mode: 'historical', samples, hash };
}

function loadDecisions(): Decision[] {
  const filePath = path.join(__dirname, 'profiles_decisions.jsonl');
  const lines = fs.readFileSync(filePath, 'utf-8').split('\n').filter(Boolean);
  return lines.map((line) => JSON.parse(line));
}

function getLikesWithComments(decisions: Decision[]): Decision[] {
  return decisions.filter(
    (d) => d.decision === 'like' && d.comment && d.comment.trim().length > 0
  );
}

function samplePerModel(
  likes: Decision[],
  samplesPerModel: number
): Map<string, Decision[]> {
  const byModel = new Map<string, Decision[]>();
  for (const d of likes) {
    const existing = byModel.get(d.model) || [];
    existing.push(d);
    byModel.set(d.model, existing);
  }

  // Only include models with meaningful sample sizes (>= 10 likes)
  const sampled = new Map<string, Decision[]>();
  for (const [model, entries] of byModel) {
    if (entries.length < 10) continue;
    const shuffled = entries.sort(() => Math.random() - 0.5);
    sampled.set(model, shuffled.slice(0, samplesPerModel));
  }
  return sampled;
}

function buildUserPrompt(d: Decision): string {
  return `Profile: ${d.profile.firstName}, ${d.profile.age}
Prompts on their profile:
${d.prompts.map((p) => `  "${p.question}" → "${p.answer}"`).join('\n')}

${d.decisionSource === 'image' ? '(This opener was sent on a photo, not a text prompt)\n' : ''}Opener message sent: "${d.comment}"`;
}

function extractAndParseJSON(text: string): any {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    // Extract JSON from first { to last }
    const firstBrace = trimmed.indexOf('{');
    const lastBrace = trimmed.lastIndexOf('}');
    if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
      throw new Error('No valid JSON object found');
    }
    return JSON.parse(trimmed.substring(firstBrace, lastBrace + 1));
  }
}

async function judgeOpener(d: Decision): Promise<JudgeScores | null> {
  try {
    const systemMsg: ChatCompletionMessageParam = {
      role: 'system',
      content: [{
        type: 'text',
        text: JUDGE_SYSTEM_PROMPT,
        cache_control: { type: 'ephemeral' },
      }] as any,
    };

    const content = await llmChat(
      buildUserPrompt(d),
      [systemMsg],
      JUDGE_MODEL,
      'json_object',
    );

    const parsed = extractAndParseJSON(content);
    if (
      parsed.reply_likelihood?.score != null &&
      parsed.humor_quality?.score != null &&
      parsed.creativity?.score != null &&
      parsed.specificity?.score != null &&
      parsed.human_likeness?.score != null &&
      parsed.closer_strength?.score != null
    ) {
      return parsed as JudgeScores;
    }
    console.error('Invalid judge response:', JSON.stringify(parsed).slice(0, 200));
    return null;
  } catch (e: any) {
    console.error('Judge call failed:', e.message?.slice(0, 150));
    return null;
  }
}

async function runWithConcurrency<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  concurrency: number
): Promise<R[]> {
  const results: R[] = [];
  let index = 0;

  async function worker() {
    while (index < items.length) {
      const i = index++;
      results[i] = await fn(items[i]);
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()));
  return results;
}

function aggregate(results: EvalResult[]): ModelAggregate[] {
  const byModel = new Map<string, EvalResult[]>();
  for (const r of results) {
    const existing = byModel.get(r.model) || [];
    existing.push(r);
    byModel.set(r.model, existing);
  }

  const aggregates: ModelAggregate[] = [];
  for (const [model, entries] of byModel) {
    const n = entries.length;
    const avg = (dim: keyof JudgeScores) =>
      entries.reduce((s, e) => s + e.scores[dim].score, 0) / n;
    const reply = avg('reply_likelihood');
    const humor = avg('humor_quality');
    const creativity = avg('creativity');
    const specificity = avg('specificity');
    const human = avg('human_likeness');
    const closer = avg('closer_strength');
    aggregates.push({
      model,
      samples: n,
      reply_likelihood: +reply.toFixed(1),
      humor_quality: +humor.toFixed(1),
      creativity: +creativity.toFixed(1),
      specificity: +specificity.toFixed(1),
      human_likeness: +human.toFixed(1),
      closer_strength: +closer.toFixed(1),
      avg: +((reply + humor + creativity + specificity + human + closer) / 6).toFixed(1),
    });
  }

  return aggregates.sort((a, b) => b.avg - a.avg);
}

function printReport(aggregates: ModelAggregate[]) {
  const header = 'Model                              | Samples | Reply | Humor | Creat | Specf | Human | Close | Avg';
  const sep = '-----------------------------------|---------|-------|-------|-------|-------|-------|-------|-----';
  console.log('\n' + header);
  console.log(sep);
  for (const a of aggregates) {
    const model = a.model.padEnd(35);
    const samples = String(a.samples).padStart(5);
    const reply = a.reply_likelihood.toFixed(1).padStart(5);
    const humor = a.humor_quality.toFixed(1).padStart(5);
    const creat = a.creativity.toFixed(1).padStart(5);
    const specf = a.specificity.toFixed(1).padStart(5);
    const human = a.human_likeness.toFixed(1).padStart(5);
    const close = a.closer_strength.toFixed(1).padStart(5);
    const avg = a.avg.toFixed(1).padStart(4);
    console.log(`${model} | ${samples}   | ${reply} | ${humor} | ${creat} | ${specf} | ${human} | ${close} | ${avg}`);
  }
  console.log();
}

function printCostSummary() {
  const summary = getCostSummary();
  const cacheHitRate =
    summary.cacheReadTokens + summary.cacheWriteTokens > 0
      ? (
          (summary.cacheReadTokens /
            (summary.cacheReadTokens + summary.cacheWriteTokens)) *
          100
        ).toFixed(1)
      : '0.0';

  console.log('--- Cost Summary (All) ---');
  console.log(`  LLM calls:         ${summary.calls}`);
  console.log(`  Input tokens:      ${summary.inputTokens.toLocaleString()}`);
  console.log(`  Output tokens:     ${summary.outputTokens.toLocaleString()}`);
  console.log(`  Cache write:       ${summary.cacheWriteTokens.toLocaleString()}`);
  console.log(`  Cache read:        ${summary.cacheReadTokens.toLocaleString()}`);
  console.log(`  Cache hit rate:    ${cacheHitRate}%`);
  console.log(`  Total cost:        $${summary.totalCost.toFixed(4)}`);
}

// --- Historical Mode ---

async function runHistoricalMode(args: { samples: number; hash?: string }) {
  console.log(`Opener Quality Eval — Historical Mode — ${args.samples} samples per model`);
  if (args.hash) console.log(`Filtering by prompt hash prefix: ${args.hash}`);
  console.log(`Judge model: ${JUDGE_MODEL} (Anthropic direct)\n`);

  const decisions = loadDecisions();
  let likes = getLikesWithComments(decisions);
  console.log(`Total decisions: ${decisions.length}, likes with comments: ${likes.length}`);

  if (args.hash) {
    const before = likes.length;
    likes = likes.filter((d) => d.openerPromptHash?.startsWith(args.hash!));
    console.log(`Hash filter "${args.hash}": ${likes.length}/${before} records match`);
  }

  const sampled = samplePerModel(likes, args.samples);
  for (const [model, entries] of sampled) {
    console.log(`  ${model}: ${entries.length} sampled`);
  }

  const allToEval: Decision[] = [];
  for (const entries of sampled.values()) {
    allToEval.push(...entries);
  }
  console.log(`\nTotal openers to evaluate: ${allToEval.length}`);
  console.log(`Running evals (concurrency: ${CONCURRENCY})...\n`);

  let completed = 0;
  const judgeResults = await runWithConcurrency(
    allToEval,
    async (d) => {
      const scores = await judgeOpener(d);
      completed++;
      if (completed % 10 === 0 || completed === allToEval.length) {
        process.stdout.write(`  ${completed}/${allToEval.length} evaluated\n`);
      }
      return { decision: d, scores };
    },
    CONCURRENCY
  );

  const evalResults: EvalResult[] = [];
  for (const { decision, scores } of judgeResults) {
    if (!scores) continue;
    evalResults.push({
      model: decision.model,
      comment: decision.comment!,
      promptQuestion: decision.prompts.map((p) => p.question).join(' | '),
      promptAnswer: decision.prompts.map((p) => p.answer).join(' | '),
      profileName: decision.profile.firstName,
      profileAge: decision.profile.age,
      decisionSource: decision.decisionSource,
      scores,
    });
  }

  console.log(`\nSuccessful evals: ${evalResults.length}/${allToEval.length}`);

  const aggregates = aggregate(evalResults);
  printReport(aggregates);
  printCostSummary();

  const outputPath = path.join(__dirname, 'eval_results.json');
  const output = {
    metadata: {
      mode: 'historical' as const,
      timestamp: new Date().toISOString(),
      judgeModel: JUDGE_MODEL,
      samplesPerModel: args.samples,
      hashFilter: args.hash ?? null,
      totalEvals: evalResults.length,
      cost: getCostSummary().totalCost,
      tokens: getCostSummary(),
    },
    aggregates,
    results: evalResults,
  };
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nDetailed results written to: ${outputPath}`);

  // Append to persistent benchmark log (JSONL)
  const benchmarkPath = path.join(__dirname, 'eval_benchmarks.jsonl');
  const finalCost = getCostSummary();
  for (const agg of aggregates) {
    const entry = {
      timestamp: new Date().toISOString(),
      mode: 'historical',
      model: agg.model,
      hashFilter: args.hash ?? null,
      samples: agg.samples,
      reply_likelihood: agg.reply_likelihood,
      humor_quality: agg.humor_quality,
      creativity: agg.creativity,
      specificity: agg.specificity,
      human_likeness: agg.human_likeness,
      closer_strength: agg.closer_strength,
      avg: agg.avg,
      cost: finalCost.totalCost,
      judgeModel: JUDGE_MODEL,
    };
    fs.appendFileSync(benchmarkPath, JSON.stringify(entry) + '\n');
  }
  console.log(`Benchmark appended to: ${benchmarkPath}`);
}

// --- Generate Mode ---

async function runGenerateMode(args: { samples: number; models: string[]; prompts?: string[]; bestOf?: number }) {
  // Resolve prompt variants
  const promptVariants: { label: string; filePath: string }[] = [];
  if (args.prompts && args.prompts.length > 0) {
    for (const p of args.prompts) {
      const resolved = path.resolve(p);
      if (!fs.existsSync(resolved)) {
        console.error(`Prompt file not found: ${resolved}`);
        process.exit(1);
      }
      promptVariants.push({ label: path.basename(p, path.extname(p)), filePath: resolved });
    }
  } else {
    // Default: use the current opener_prompt.yaml
    promptVariants.push({ label: 'current', filePath: '' });
  }

  const combos = promptVariants.length * args.models.length;
  console.log(`Opener Quality Eval — Generate Mode`);
  console.log(`  ${args.samples} profiles × ${args.models.length} model(s) × ${promptVariants.length} prompt(s) = ${combos} combos`);
  console.log(`  Models: ${args.models.join(', ')}`);
  if (args.prompts) console.log(`  Prompts: ${promptVariants.map((p) => p.label).join(', ')}`);
  if (args.bestOf && args.bestOf > 1) console.log(`  Best-of-${args.bestOf} generation enabled`);
  console.log(`  Judge model: ${JUDGE_MODEL} (Anthropic direct)\n`);

  // Load profiles: deduplicate by userId, take N unique profiles with text prompts
  const decisions = loadDecisions();
  const likes = getLikesWithComments(decisions).filter((d) => d.decisionSource === 'text');
  const seenUsers = new Set<string>();
  const uniqueProfiles: Decision[] = [];
  const shuffled = likes.sort(() => Math.random() - 0.5);
  for (const d of shuffled) {
    const uid = d.userId ?? `${d.profile.firstName}_${d.profile.age}`;
    if (seenUsers.has(uid)) continue;
    seenUsers.add(uid);
    uniqueProfiles.push(d);
    if (uniqueProfiles.length >= args.samples) break;
  }
  console.log(`Sampled ${uniqueProfiles.length} unique profiles from ${likes.length} text likes\n`);

  const provider = (process.env.LLM_PROVIDER || 'anthropic') as any;
  setLLMProvider(provider);
  console.log(`  Generation provider: ${provider}`);
  resetCostTracker();

  interface GeneratedOpener {
    model: string;
    promptLabel: string;
    promptHash: string;
    profileDecision: Decision;
    promptQuestion: string;
    promptAnswer: string;
    openerText: string;
  }

  const allGenerated: GeneratedOpener[] = [];
  const promptHashes: Map<string, string> = new Map();

  // Pipeline: judge openers as they're generated
  const judgeResultsMap = new Map<GeneratedOpener, { scores: JudgeScores | null }>();
  const judgeQueue: GeneratedOpener[] = [];
  let judgeWorkersDone = false;

  function judgeOneOpener(gen: GeneratedOpener): Promise<{ scores: JudgeScores | null }> {
    const syntheticDecision: Decision = {
      timestamp: new Date().toISOString(),
      decision: 'like',
      decisionSource: 'text',
      comment: gen.openerText,
      prompts: [{ question: gen.promptQuestion, answer: gen.promptAnswer }],
      profile: gen.profileDecision.profile,
      model: gen.model,
    };
    return judgeOpener(syntheticDecision).then((scores) => ({ scores }));
  }

  // Start background judge workers that drain the queue while generation runs
  const backgroundJudgePromise = (async () => {
    const activeJudges: Promise<void>[] = [];
    const processOne = async (gen: GeneratedOpener) => {
      const result = await judgeOneOpener(gen);
      judgeResultsMap.set(gen, result);
      const done = judgeResultsMap.size;
      if (done % 10 === 0) {
        process.stdout.write(`  [pipeline] ${done} judged so far\n`);
      }
    };

    while (!judgeWorkersDone || judgeQueue.length > 0) {
      while (judgeQueue.length > 0 && activeJudges.length < CONCURRENCY) {
        const gen = judgeQueue.shift()!;
        const p = processOne(gen).then(() => {
          activeJudges.splice(activeJudges.indexOf(p), 1);
        });
        activeJudges.push(p);
      }
      if (activeJudges.length > 0) {
        await Promise.race(activeJudges);
      } else {
        await new Promise((r) => setTimeout(r, 50));
      }
    }
    await Promise.all(activeJudges);
  })();

  for (const variant of promptVariants) {
    // Set prompt override (or clear for default)
    if (variant.filePath) {
      const text = fs.readFileSync(variant.filePath, 'utf-8');
      setOpenerPromptOverride(text);
    } else {
      setOpenerPromptOverride(null);
    }
    const promptMeta = await getOpenerPromptMetadata();
    promptHashes.set(variant.label, promptMeta.hash);
    console.log(`Prompt "${variant.label}": hash ${promptMeta.hash.slice(0, 8)}`);

    for (const model of args.models) {
      const comboLabel = promptVariants.length > 1 ? `${variant.label} / ${model}` : model;
      console.log(`\nGenerating: ${comboLabel}...`);

      const genTasks: { profile: Decision; promptIdx: number; id: string; profilePrompt: string }[] = [];
      for (const profile of uniqueProfiles) {
        for (let pi = 0; pi < profile.prompts.length; pi++) {
          const p = profile.prompts[pi];
          const id = `${profile.userId ?? profile.profile.firstName}_p${pi}`;
          genTasks.push({
            profile,
            promptIdx: pi,
            id,
            profilePrompt: `"${p.question}" → "${p.answer}"`,
          });
        }
      }

      let genCompleted = 0;
      await runWithConcurrency(
        uniqueProfiles,
        async (profile) => {
          const profileTasks = genTasks.filter((t) => t.profile === profile);
          const items = profileTasks.map((t) => ({ id: t.id, profilePrompt: t.profilePrompt }));

          try {
            const results = args.bestOf && args.bestOf > 1
              ? await generateOpenersFromYamlBatchBestOfN(items, profile.profile.firstName, model, args.bestOf)
              : await generateOpenersFromYamlBatch(items, profile.profile.firstName, model);
            for (const task of profileTasks) {
              const text = results.get(task.id);
              if (text) {
                const opener: GeneratedOpener = {
                  model: comboLabel,
                  promptLabel: variant.label,
                  promptHash: promptMeta.hash,
                  profileDecision: task.profile,
                  promptQuestion: task.profile.prompts[task.promptIdx].question,
                  promptAnswer: task.profile.prompts[task.promptIdx].answer,
                  openerText: text,
                };
                allGenerated.push(opener);
                judgeQueue.push(opener);
              }
            }
          } catch (e: any) {
            console.error(`  Generation failed for ${profile.profile.firstName}: ${e.message?.slice(0, 100)}`);
          }
          genCompleted++;
          if (genCompleted % 5 === 0 || genCompleted === uniqueProfiles.length) {
            process.stdout.write(`  ${genCompleted}/${uniqueProfiles.length} profiles done\n`);
          }
        },
        GEN_CONCURRENCY
      );

      const count = allGenerated.filter((g) => g.model === comboLabel).length;
      console.log(`  Generated ${count} openers`);
    }
  }

  // Clear override after generation
  setOpenerPromptOverride(null);

  // Capture generation cost
  const genCostSummary = getCostSummary();
  console.log(`\nGeneration complete. ${allGenerated.length} total openers.`);
  console.log(`Generation cost: $${genCostSummary.totalCost.toFixed(4)}`);

  // Signal background judges that no more openers are coming, then wait
  judgeWorkersDone = true;
  console.log(`Waiting for ${allGenerated.length - judgeResultsMap.size} remaining judge calls...\n`);
  await backgroundJudgePromise;

  console.log(`  ${judgeResultsMap.size}/${allGenerated.length} judged`);

  const evalResults: EvalResult[] = [];
  for (const [gen, result] of judgeResultsMap) {
    if (!result.scores) continue;
    evalResults.push({
      model: gen.model,
      comment: gen.openerText,
      promptQuestion: gen.promptQuestion,
      promptAnswer: gen.promptAnswer,
      profileName: gen.profileDecision.profile.firstName,
      profileAge: gen.profileDecision.profile.age,
      decisionSource: 'text',
      scores: result.scores,
    });
  }

  console.log(`\nSuccessful evals: ${evalResults.length}/${allGenerated.length}`);

  const aggregates = aggregate(evalResults);
  printReport(aggregates);
  printCostSummary();

  // Write results with timestamp
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const outputPath = path.join(__dirname, `eval_results_gen_${ts}.json`);
  const finalCost = getCostSummary();
  const output = {
    metadata: {
      mode: 'generate' as const,
      timestamp: new Date().toISOString(),
      judgeModel: JUDGE_MODEL,
      models: args.models,
      promptVariants: Object.fromEntries(promptHashes),
      samplesPerModel: args.samples,
      totalGenerated: allGenerated.length,
      totalEvals: evalResults.length,
      totalCost: finalCost.totalCost,
    },
    aggregates,
    results: evalResults,
  };
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nDetailed results written to: ${outputPath}`);

  // Append to persistent benchmark log (JSONL)
  const benchmarkPath = path.join(__dirname, 'eval_benchmarks.jsonl');
  for (const agg of aggregates) {
    const entry = {
      timestamp: new Date().toISOString(),
      mode: 'generate',
      model: agg.model,
      promptVariants: Object.fromEntries(promptHashes),
      samples: agg.samples,
      reply_likelihood: agg.reply_likelihood,
      humor_quality: agg.humor_quality,
      creativity: agg.creativity,
      specificity: agg.specificity,
      human_likeness: agg.human_likeness,
      closer_strength: agg.closer_strength,
      avg: agg.avg,
      cost: finalCost.totalCost,
      judgeModel: JUDGE_MODEL,
    };
    fs.appendFileSync(benchmarkPath, JSON.stringify(entry) + '\n');
  }
  console.log(`Benchmark appended to: ${benchmarkPath}`);
}

// --- Main ---

async function main() {
  const args = parseArgs();
  if (args.mode === 'generate') {
    await runGenerateMode(args);
  } else {
    await runHistoricalMode(args);
  }
}

main()
  .catch((e) => {
    console.error('Fatal error:', e);
    process.exit(1);
  })
  .finally(() => {
    cleanupLLMProvider();
  });
