import { deepseek } from '@ai-sdk/deepseek';
import { streamText } from 'ai';
import { run } from '../../lib/run';

run(async () => {
  const model = deepseek('deepseek-v4-flash');
  const userMessage = 'Tell me a short story about a dragon in one sentence.';

  console.log('=== TURN 1 ===\n');
  const result1 = streamText({
    model,
    messages: [{ role: 'user', content: userMessage }],
  });

  for await (const part of result1.fullStream) {
    if (part.type === 'reasoning-delta') {
      process.stdout.write(`\x1b[34m${part.text}\x1b[0m`);
    } else if (part.type === 'text-delta') {
      process.stdout.write(part.text);
    }
  }

  const turn1Messages = (await result1.response).messages;

  console.log('\n\n--- Turn 1 response messages ---');
  console.log(JSON.stringify(turn1Messages, null, 2));

  console.log('\n=== TURN 2 ===\n');
  const result2 = streamText({
    model,
    messages: [
      { role: 'user', content: userMessage },
      ...turn1Messages,
      { role: 'user', content: 'Now make it rhyme.' },
    ],
  });

  for await (const part of result2.fullStream) {
    if (part.type === 'reasoning-delta') {
      process.stdout.write(`\x1b[34m${part.text}\x1b[0m`);
    } else if (part.type === 'text-delta') {
      process.stdout.write(part.text);
    }
  }

  console.log('\n');
  console.log('Usage:', await result2.usage);
});
