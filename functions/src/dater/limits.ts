export function createLikeLimiter(maxLikes: number) {
  let remaining = Math.max(0, Math.floor(maxLikes));
  let used = 0;
  return {
    tryTake(): boolean {
      if (remaining <= 0) return false;
      remaining -= 1;
      used += 1;
      return true;
    },
    release(): void {
      if (used > 0) {
        used -= 1;
        remaining += 1;
      }
    },
    isExhausted(): boolean {
      return remaining <= 0;
    },
    getUsed(): number {
      return used;
    },
  };
}

export function createCancelToken() {
  let aborted = false;
  let reason: unknown = undefined;
  return {
    abort(err: unknown) {
      if (!aborted) {
        aborted = true;
        reason = err;
      }
    },
    isAborted() {
      return aborted;
    },
    reason() {
      return reason;
    },
  };
}

export function createHingeRequestLimiter(cancelToken: { abort(err: unknown): void }, max = 2000) {
  let remaining = max;
  return {
    tryConsume(count = 1) {
      if (remaining < count) {
        const err = new Error(`Hinge request limit (${max}) reached`);
        cancelToken.abort(err);
        throw err;
      }
      remaining -= count;
    },
    remaining() {
      return remaining;
    },
  };
}

