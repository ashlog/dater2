export function timeout(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function dateDifferenceFormatted(date1: Date, date2: Date) {
  // Get the timestamps of the two dates in milliseconds
  const time1 = date1.getTime();
  const time2 = date2.getTime();

  // Find the difference between the timestamps
  let diffMilliseconds = Math.abs(time2 - time1);

  // Calculate days, hours, and minutes
  const days = Math.floor(diffMilliseconds / (1000 * 60 * 60 * 24));

  if (days > 0) {
    return `${days} day${days > 1 ? 's' : ''} ago`;
  }

  diffMilliseconds -= days * 1000 * 60 * 60 * 24;
  const hours = Math.floor(diffMilliseconds / (1000 * 60 * 60));

  if (hours > 0) {
    return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  }

  diffMilliseconds -= hours * 1000 * 60 * 60;
  const minutes = Math.floor(diffMilliseconds / (1000 * 60));

  return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
}
