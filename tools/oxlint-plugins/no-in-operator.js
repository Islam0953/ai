const rule = {
  create(context) {
    return {
      BinaryExpression(node) {
        if (node.operator !== 'in') return;
        context.report({
          node,
          message:
            'Use `Object.hasOwn(obj, prop)` instead of the `in` operator. `in` also walks the prototype chain, which is rarely the intended runtime check.',
        });
      },
    };
  },
};

module.exports = {
  meta: { name: 'ai-sdk-local' },
  rules: { 'no-in-operator': rule },
};
