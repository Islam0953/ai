/**
 * Augments the global `Object.hasOwn` signature with a type predicate so that
 * it narrows like the `in` operator. This makes call-site code such as
 *
 *     const enumValues = Object.hasOwn(options, 'enum') ? options.enum : undefined;
 *
 * type-check without explicit casts. The runtime behavior is unchanged.
 *
 * Narrowing rules in the truthy branch:
 * - If `T` is a union and at least one member declares `K` as a known key,
 *   narrow to those members.
 * - Otherwise, if `K` is a literal property key, intersect `T` with
 *   `{ [K]: unknown }` so that `obj[K]` is accessible.
 * - Types with a string index signature (e.g. `Record<string, X>`) are left
 *   alone — every key is technically in `keyof`, so narrowing would be a
 *   no-op anyway.
 * - When `K` is the wide `string` type, `T` is left alone too — we cannot
 *   meaningfully add an index signature without breaking the static type.
 */
declare global {
  interface ObjectConstructor {
    hasOwn<T, K extends PropertyKey>(o: T, v: K): o is HasOwnNarrow<T, K>;
  }
}

type HasOwnNarrow<T, K extends PropertyKey> = [HasOwnFilter<T, K>] extends [
  never,
]
  ? string extends K
    ? T
    : T & { [P in K]: unknown }
  : HasOwnFilter<T, K>;

type HasOwnFilter<T, K extends PropertyKey> = T extends unknown
  ? string extends keyof T
    ? never
    : K extends keyof T
      ? T
      : never
  : never;

export {};
