// Define a type for the asynchronous comparison function
type CompareFunction<T> = (a: T, b: T) => Promise<number>;

export async function asyncQuickSort<T>(
  arr: T[],
  left = 0,
  right: number = arr.length - 1,
  compareAsync: CompareFunction<T>
): Promise<T[]> {
  if (left < right) {
    const pivotIndex = await partition(arr, left, right, compareAsync);
    await Promise.all([
      asyncQuickSort(arr, left, pivotIndex - 1, compareAsync),
      asyncQuickSort(arr, pivotIndex + 1, right, compareAsync),
    ]);
  }
  return arr;
}

async function partition<T>(
  arr: T[],
  left: number,
  right: number,
  compareAsync: CompareFunction<T>
): Promise<number> {
  const pivot = arr[right];
  let i = left;
  for (let j = left; j < right; j++) {
    if ((await compareAsync(arr[j], pivot)) < 0) {
      [arr[i], arr[j]] = [arr[j], arr[i]];
      i++;
    }
  }
  [arr[i], arr[right]] = [arr[right], arr[i]];
  return i;
}
