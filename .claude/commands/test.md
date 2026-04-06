Generate comprehensive Jest unit tests for the following code.

For each exported function/class, write tests covering:
1. Happy path — normal expected usage
2. Edge cases — empty inputs, boundary values, null/undefined
3. Error cases — invalid inputs, expected throws
4. Side effects — state mutations, mocked I/O

Use `describe`/`it` blocks with descriptive names, `jest.mock()` for external dependencies, and `beforeEach`/`afterEach` for state reset.

Output only the test file sand save it undet __test__ folder

$ARGUMENTS
