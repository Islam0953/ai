import * as z4 from 'zod/v4';
import * as z3 from 'zod/v3';
import { run } from '../lib/run';

run(async () => {
  const z4Schema = z4.object({
    name: z4.string(),
    age: z4.number(),
  });

  const z3Schema = z3.object({
    name: z3.string(),
    age: z3.number(),
  });

  const z4StandardSchema = z4Schema['~standard'];
  const z3StandardSchema = z3Schema['~standard'];

  console.log(JSON.stringify(z4Schema, null, 2));
  console.log(Object.hasOwn(z4Schema, '_zod'));
  console.log(JSON.stringify(z3Schema, null, 2));
  console.log(Object.hasOwn(z3Schema, '_zod'));

  console.log(z4StandardSchema.vendor);
  console.log(z3StandardSchema.vendor);
});
