Refactor the following code applying these improvements:

1. **Modern syntax** — `const`/`let`, arrow functions, template literals, optional chaining
2. **Array methods** — replace `for` loops with `map`/`filter`/`find`/`reduce` where clearer
3. **Error handling** — wrap I/O and parsing in try/catch with meaningful messages
4. **Security** — fix plaintext secrets, use `===`, use `crypto` for random tokens
5. **Immutability** — avoid mutating inputs; return new objects/arrays
6. **Readability** — meaningful names, consistent style, remove dead code

Output format:
1. `## What Changed` — bullet list of key transformations
2. Complete refactored code block

Preserve the same public API (same exports and function signatures) unless a change is essential for correctness.

$ARGUMENTS
