import { asyncQuickSort } from './async_sort';
import { compare } from './openai';

export async function llmSort(arr: string[]) {
  const newArr = await asyncQuickSort(
    arr,
    0,
    arr.length - 1,
    async (a: any, b: any) => {
      const best = await compare([a, b]);
      return best === a ? -1 : 1;
    }
  );
  return newArr;
}

