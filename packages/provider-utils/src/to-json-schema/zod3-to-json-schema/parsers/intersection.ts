import { ZodIntersectionDef } from 'zod/v3';
import { parseDef } from '../parse-def';
import { JsonSchema7Type } from '../parse-types';
import { Refs } from '../refs';
import { JsonSchema7StringType } from './string';

export type JsonSchema7AllOfType = {
  allOf: JsonSchema7Type[];
  unevaluatedProperties?: boolean;
};

const isJsonSchema7AllOfType = (
  type: JsonSchema7Type | JsonSchema7StringType,
): type is JsonSchema7AllOfType => {
  if (Object.hasOwn(type, 'type') && type.type === 'string') return false;
  return Object.hasOwn(type, 'allOf');
};

export function parseIntersectionDef(
  def: ZodIntersectionDef,
  refs: Refs,
): JsonSchema7AllOfType | JsonSchema7Type | undefined {
  const allOf = [
    parseDef(def.left._def, {
      ...refs,
      currentPath: [...refs.currentPath, 'allOf', '0'],
    }),
    parseDef(def.right._def, {
      ...refs,
      currentPath: [...refs.currentPath, 'allOf', '1'],
    }),
  ].filter((x): x is JsonSchema7Type => !!x);

  const mergedAllOf: JsonSchema7Type[] = [];
  // If either of the schemas is an allOf, merge them into a single allOf
  allOf.forEach(schema => {
    if (isJsonSchema7AllOfType(schema)) {
      mergedAllOf.push(...schema.allOf);
    } else {
      let nestedSchema: JsonSchema7Type = schema;
      if (
        Object.hasOwn(schema, 'additionalProperties') &&
        schema.additionalProperties === false
      ) {
        const { additionalProperties: _additionalProperties, ...rest } =
          schema as Record<string, any>;
        nestedSchema = rest as JsonSchema7Type;
      }
      mergedAllOf.push(nestedSchema);
    }
  });
  return mergedAllOf.length ? { allOf: mergedAllOf } : undefined;
}
