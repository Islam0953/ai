import { STREAMABLE_VALUE_TYPE, StreamableValue } from './streamable-value';

export function isStreamableValue(value: unknown): value is StreamableValue {
  return (
    value != null &&
    typeof value === 'object' &&
    Object.hasOwn(value, 'type') &&
    value.type === STREAMABLE_VALUE_TYPE
  );
}
