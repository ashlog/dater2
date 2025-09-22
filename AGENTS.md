# Agent/Runner Documentation

## Overview

The runner system in `functions/src/dater/runner.ts` manages the automated profile processing, liking, and decision-making workflow for the Hinge dating app automation.

## Architecture

### Main Components

1. **run()** - Main entry point that processes multiple location settings
2. **fetchProfilesForSetting()** - Fetches profiles from cache or Hinge API
3. **processProfileEntry()** - Processes individual profiles (validation, generation, decision)
4. **likeWithImage()** - Sends likes with photo comments
5. **likeWithText()** - Sends likes with text prompt responses
6. **validateProfileImages()** - Validates profile photos using AI scoring

## Flow Control

### Iteration Strategy

The system uses a `while` loop (not a fixed iteration count) to continue processing until the like limit is reached:

```typescript
while (!likeLimiter.isExhausted() && !cancelToken.isAborted()) {
  // Fetch profiles
  // Process profiles
  // Continue until maxLikes reached
}
```

This ensures the full requested number of likes (e.g., 300) are attempted, rather than stopping after a fixed number of batches.

### Location Processing

The system processes multiple geographic locations sequentially:
1. Process profiles from cache if available
2. Fetch fresh profiles from Hinge when cache is exhausted
3. Move to next location when no more profiles available
4. Stop when like limit reached OR all locations exhausted

## Profile Visited State Management

### When Profiles are Marked as Visited

Profiles are marked as "visited" (won't be processed again) in these cases:

#### ✅ Always Mark as Visited:
1. **Successful likes** - The like was sent successfully to Hinge
2. **Successful skips** - Profile was intentionally skipped (failed image validation)
3. **400 errors** - Invalid request errors (profile already liked, doesn't exist, etc.)
   - No point retrying these - they're permanent failures

#### ❌ Leave Unvisited (Available for Retry):
1. **Transient errors** - Network timeouts, 5xx errors, temporary API issues
2. **Generation failures** - AI/OpenRouter failures (with consecutive failure limit)
3. **Like limit reached** - Profile never got processed due to limit

### Implementation

```typescript
// In likeWithImage() and likeWithText():
if (!error) {
  // Success - mark as visited
  await markVisited(undefined, locKey, ratingToken);
} else {
  const errorInfo = formatDeliveryError(error);
  const is400 = errorInfo.code === '400' || errorInfo.message.includes('400');

  if (is400) {
    // Permanent failure - mark as visited
    await markVisited(undefined, locKey, ratingToken);
  } else {
    // Transient error - leave unvisited for retry
    // Profile will be retried in next run
  }
}
```

### Rationale

This strategy ensures:
- **Resilience**: Temporary network issues don't cause profile loss
- **Efficiency**: Invalid profiles (400s) aren't retried endlessly
- **Completeness**: System can recover from transient failures automatically
- **Progress**: Failed profiles are retried in subsequent runs

## Error Handling Strategy

### Non-Fatal Errors (Continue Processing)

These errors are logged but don't stop the run:
- Individual profile like failures (unless rate limited)
- Image scoring/generation errors
- Batch processing errors
- Location fetch errors

### Fatal Errors (Stop Processing)

These errors abort the entire run:
- Rate limiter exhaustion (2000 Hinge API requests)
- Consecutive generation failures (5 in a row) - indicates OpenRouter outage
- Cancel token abort (triggered by fatal errors)

### Error Propagation

```typescript
// OLD (too aggressive):
catch (e) {
  cancelToken.abort(e);  // ❌ Aborts entire run
  throw e;               // ❌ Breaks batch processing
}

// NEW (resilient):
catch (e) {
  // Only abort on truly fatal errors
  if (isFatalError(e)) {
    cancelToken.abort(e);
    throw e;
  }
  // Log and continue for transient errors
  console.error('Error:', e);
  // Continue processing other profiles
}
```

## Limiters and Concurrency

### Like Limiter
- Tracks number of likes sent
- Reserves slots before expensive processing
- Releases slots if profile is skipped or errors occur
- Prevents exceeding requested like count

### Hinge Request Limiter
- Tracks API requests to Hinge
- Hard limit of 2000 requests per run
- Prevents hitting API rate limits
- Fatal error when exhausted

### Concurrency
- Processes 10 profiles in parallel (via `pLimit`)
- Balances speed with resource usage
- Each profile reserves a like slot before processing

## Logging

### Key Log Patterns

```
[FETCH] Location {key}: {count} unvisited profiles in cache
[FETCH] Fetching fresh recommendations from Hinge for location {key}
[FETCH] Returning {count} profiles for processing
[LOCATION] No more profiles available for location: {key}, moving to next location
[ERROR] Error processing profiles batch: {error}
[TIMING] Profile processing times for performance monitoring
```

### Status Messages

- Like success: `Likes {count} Like Rate: {rate}%`
- Like failed (400): `Like failed with 400 (invalid request), marking as visited`
- Like failed (transient): `Like failed with transient error, profile left unvisited for retry`

## Consecutive Failure Protection

To prevent infinite loops when AI services fail:

```typescript
if (allCandidates.length === 0) {
  ctx.failureCounter.consecutive++;

  if (ctx.failureCounter.consecutive >= MAX_CONSECUTIVE_FAILURES) {
    console.error('FATAL: 5 consecutive generation failures detected');
    process.exit(1);  // Stop to prevent wasted resources
  }
}
```

## Best Practices

1. **Always check cache first** before fetching from Hinge API
2. **Release like slots** when profiles are skipped or fail
3. **Don't mark as visited** for transient errors
4. **Mark as visited** for permanent failures (400s)
5. **Log extensively** for debugging and monitoring
6. **Handle errors gracefully** - continue processing when possible
7. **Use cancel token** only for truly fatal errors

## Configuration

Key constants in `runner.ts`:
- `concurrency = 10` - Parallel profile processing
- `MAX_CONSECUTIVE_FAILURES = 5` - Generation failure threshold
- `model = 'anthropic/claude-sonnet-4.5'` - AI model for generations
- Hinge request limit: 2000 per run

## Future Improvements

Potential enhancements:
- Exponential backoff for transient errors
- More granular error categorization (401, 403, 429, etc.)
- Metrics collection and reporting
- Configurable retry limits per profile
- Profile quality scoring and filtering
