export function isAsyncGenerator<T, TReturn, TNext>(
  value: unknown,
): value is AsyncGenerator<T, TReturn, TNext> {
  return (
    value != null &&
    typeof value === 'object' &&
    typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === 'function'
  );
}
