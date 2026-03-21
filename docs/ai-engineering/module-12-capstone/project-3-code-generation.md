---
sidebar_position: 4
title: "Project 3: Code Generation Assistant"
description: Build a coding assistant that generates code from natural language specs, self-reviews its output for bugs, and iterates based on test results.
---

# Project 3: Code Generation Assistant

**Difficulty:** Advanced
**Modules:** 2, 3, 6, 8, 11
**Time:** 12–16 hours

---

## Problem Statement

Build a code generation assistant that:
1. Accepts a natural language specification and optional existing codebase context
2. Generates Python code that fulfills the specification
3. Self-reviews the generated code for bugs, edge cases, and style issues
4. Generates a test suite for the code
5. Iterates — fixes bugs identified in the self-review or test failures

**What makes this hard**: Evaluation. Unlike most LLM tasks, code has an objective oracle: does it run? Do the tests pass? This makes the evaluation loop unusually clean — and makes the system self-correcting in a way that text generation cannot be.

---

## Architecture

```
Natural language spec
        ↓
[Spec Parser]
  Extract: function name, inputs, outputs, constraints, examples
        ↓
[Code Generator]
  Generate function implementation
        ↓
[Self-Reviewer]
  LLM reviews own code for:
  - Logic bugs
  - Edge cases (empty input, None, overflow)
  - Style and correctness
        ↓
[Bug Fixed Code] (if issues found)
        ↓
[Test Generator]
  Generate pytest test suite based on spec
        ↓
[Test Runner]
  Execute tests in sandbox
        ↓
  All pass? → Return code + tests
  Failures? → [Bug Fixer] → back to Test Runner
              (max 3 iterations)
        ↓
Final: code + tests + iteration history
```

---

## Implementation Guide

### Phase 1: Spec Parsing (1 hour)

```python
import os
import json
import subprocess
import tempfile
from pathlib import Path
import openai

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SPEC_PARSER_PROMPT = """Parse this natural language specification into a structured format.

Specification: {spec}

Return JSON:
{{
  "function_name": "snake_case_name",
  "description": "one-line description",
  "parameters": [
    {{"name": "param_name", "type": "type_hint", "description": "..."}}
  ],
  "returns": {{"type": "type_hint", "description": "..."}},
  "constraints": ["list of constraints or edge cases to handle"],
  "examples": [
    {{"input": {{}}, "output": "expected_output"}}
  ]
}}"""


def parse_spec(specification: str) -> dict:
    """Parse a natural language spec into structured form."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": SPEC_PARSER_PROMPT.format(spec=specification)
        }],
        response_format={"type": "json_object"},
        max_tokens=600
    )
    return json.loads(response.choices[0].message.content)
```

### Phase 2: Code Generation (1–2 hours)

```python
CODE_GEN_PROMPT = """Generate a Python function implementation for the following specification.

Specification:
{spec_json}

Requirements:
- Write clean, readable Python 3.10+ code
- Include type hints for all parameters and return value
- Handle all constraints and edge cases listed in the spec
- Do not include tests in this file — only the implementation
- Add a brief docstring

Return ONLY the Python code, no explanation, no markdown fences."""


def generate_code(parsed_spec: dict) -> str:
    """Generate code from a parsed specification."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": CODE_GEN_PROMPT.format(
                spec_json=json.dumps(parsed_spec, indent=2)
            )
        }],
        max_tokens=1000,
        temperature=0.2  # Low temperature for more consistent code
    )
    code = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
    return code.strip()
```

### Phase 3: Self-Review (1–2 hours)

```python
SELF_REVIEW_PROMPT = """Review this Python code for bugs and issues.

Original specification:
{spec_json}

Generated code:
```python
{code}
```

Identify:
1. Logic bugs (incorrect behavior)
2. Missing edge case handling
3. Type errors or incorrect type handling
4. Performance issues for large inputs
5. Security issues (if applicable)

Return JSON:
{{
  "has_issues": true/false,
  "issues": [
    {{
      "severity": "CRITICAL | HIGH | MEDIUM | LOW",
      "location": "line number or function name",
      "description": "what's wrong",
      "fix": "how to fix it"
    }}
  ],
  "overall_assessment": "PASS | NEEDS_MINOR_FIX | NEEDS_MAJOR_FIX"
}}"""


def self_review_code(code: str, parsed_spec: dict) -> dict:
    """Have the model review its own generated code."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": SELF_REVIEW_PROMPT.format(
                spec_json=json.dumps(parsed_spec, indent=2),
                code=code
            )
        }],
        response_format={"type": "json_object"},
        max_tokens=600
    )
    return json.loads(response.choices[0].message.content)


FIX_CODE_PROMPT = """Fix the following Python code based on the identified issues.

Code to fix:
```python
{code}
```

Issues to fix:
{issues}

Return ONLY the corrected Python code, no explanation."""


def fix_code(code: str, issues: list[dict]) -> str:
    """Fix code based on review issues."""
    issues_text = "\n".join(
        f"- [{i['severity']}] {i['description']} → Fix: {i['fix']}"
        for i in issues
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": FIX_CODE_PROMPT.format(code=code, issues=issues_text)
        }],
        max_tokens=1000,
        temperature=0.1
    )
    fixed = response.choices[0].message.content.strip()
    if fixed.startswith("```"):
        fixed = fixed.split("\n", 1)[1].rsplit("```", 1)[0]
    return fixed.strip()
```

### Phase 4: Test Generation and Execution (2–3 hours)

```python
TEST_GEN_PROMPT = """Generate a pytest test suite for this Python function.

Function specification:
{spec_json}

Function code:
```python
{code}
```

Generate tests that cover:
1. Happy path (normal inputs from the spec examples)
2. Edge cases (empty, None, boundary values, large inputs)
3. Error cases (invalid types, out-of-range values)
4. Each constraint listed in the specification

Return ONLY the pytest code, no explanation. Start with necessary imports."""


def generate_tests(code: str, parsed_spec: dict) -> str:
    """Generate a pytest test suite for the generated code."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": TEST_GEN_PROMPT.format(
                spec_json=json.dumps(parsed_spec, indent=2),
                code=code
            )
        }],
        max_tokens=1200,
        temperature=0.2
    )
    tests = response.choices[0].message.content.strip()
    if tests.startswith("```"):
        tests = tests.split("\n", 1)[1].rsplit("```", 1)[0]
    return tests.strip()


def run_tests_in_sandbox(code: str, tests: str) -> dict:
    """Execute tests in a temporary directory, return results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write implementation
        impl_path = Path(tmpdir) / "implementation.py"
        impl_path.write_text(code)

        # Write tests (ensure they import from implementation)
        full_tests = f"import sys\nsys.path.insert(0, '{tmpdir}')\n{tests}"
        test_path = Path(tmpdir) / "test_implementation.py"
        test_path.write_text(full_tests)

        # Run pytest
        result = subprocess.run(
            ["python", "-m", "pytest", str(test_path), "-v", "--tb=short", "--timeout=10"],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=60
        )

        # Parse results
        output = result.stdout + result.stderr
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        errors = output.count(" ERROR")

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "all_passed": result.returncode == 0,
            "output": output,
            "return_code": result.returncode
        }
```

### Phase 5: Iteration Loop (1–2 hours)

```python
BUG_FIX_FROM_TESTS_PROMPT = """Fix Python code that fails tests.

Code:
```python
{code}
```

Test failures:
{test_output}

Fix the bugs that caused the test failures.
Return ONLY the corrected Python code."""


def fix_from_test_failures(code: str, test_output: str) -> str:
    """Fix code based on test failure output."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": BUG_FIX_FROM_TESTS_PROMPT.format(
                code=code,
                test_output=test_output[:3000]  # Truncate to avoid token overflow
            )
        }],
        max_tokens=1000,
        temperature=0.1
    )
    fixed = response.choices[0].message.content.strip()
    if fixed.startswith("```"):
        fixed = fixed.split("\n", 1)[1].rsplit("```", 1)[0]
    return fixed.strip()


def generate_with_iteration(specification: str, max_iterations: int = 3) -> dict:
    """Full code generation pipeline with iterative self-correction."""
    print(f"Spec: {specification[:80]}...")
    history = []

    # Parse spec
    parsed_spec = parse_spec(specification)
    print(f"Parsed: function '{parsed_spec['function_name']}' with {len(parsed_spec['parameters'])} params")

    # Generate initial code
    code = generate_code(parsed_spec)
    history.append({"stage": "initial_generation", "code": code})

    # Self-review
    review = self_review_code(code, parsed_spec)
    history.append({"stage": "self_review", "result": review})

    if review["has_issues"] and review["overall_assessment"] != "PASS":
        critical_issues = [i for i in review["issues"] if i["severity"] in ("CRITICAL", "HIGH")]
        if critical_issues:
            print(f"  Self-review found {len(critical_issues)} critical issues — fixing...")
            code = fix_code(code, critical_issues)
            history.append({"stage": "post_review_fix", "code": code})

    # Generate tests
    tests = generate_tests(code, parsed_spec)

    # Iteration loop
    for iteration in range(max_iterations):
        test_results = run_tests_in_sandbox(code, tests)
        history.append({"stage": f"test_run_{iteration+1}", "results": test_results})

        print(f"  Iteration {iteration+1}: {test_results['passed']} passed, {test_results['failed']} failed")

        if test_results["all_passed"]:
            print(f"  All tests passed after {iteration+1} iteration(s)!")
            break

        if iteration < max_iterations - 1:
            print(f"  Fixing failures...")
            code = fix_from_test_failures(code, test_results["output"])
            history.append({"stage": f"test_fix_{iteration+1}", "code": code})
    else:
        print(f"  Warning: Tests still failing after {max_iterations} iterations")

    return {
        "specification": specification,
        "parsed_spec": parsed_spec,
        "final_code": code,
        "tests": tests,
        "test_results": test_results,
        "iterations": len([h for h in history if "test_run" in h["stage"]]),
        "history": history
    }
```

### Phase 6: Test the System

```python
# Test with increasingly complex specs
test_specs = [
    """Write a function called `is_palindrome` that takes a string and returns True
    if it's a palindrome (reads the same forwards and backwards), False otherwise.
    Ignore spaces, punctuation, and capitalization. Handle empty strings.""",

    """Write a function called `flatten_dict` that takes a nested dictionary and
    returns a flat dictionary with dot-notation keys. For example:
    {"a": {"b": {"c": 1}}} → {"a.b.c": 1}
    Handle lists as values by indexing them: {"a": [1, 2]} → {"a.0": 1, "a.1": 2}""",

    """Write a function called `rate_limiter` that implements a token bucket algorithm.
    It takes: max_tokens (int), refill_rate (float tokens/second).
    It has a method `consume(tokens)` that returns True if tokens are available
    and False if rate limit is exceeded. Thread-safe implementation required.""",
]

for spec in test_specs:
    print(f"\n{'='*60}")
    result = generate_with_iteration(spec)
    print(f"Final test results: {result['test_results']['passed']} passed, "
          f"{result['test_results']['failed']} failed")
    print(f"Total iterations: {result['iterations']}")
```

---

## Evaluation Criteria

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| Test pass rate (final) | > 80% of tests pass | Automated |
| First-attempt pass rate | > 50% pass without iteration | Track `iterations == 1` |
| Self-review bug catch rate | > 70% of real bugs caught | Compare against ground-truth bugs |
| Test coverage quality | Tests cover all spec constraints | Manual review of generated tests |
| Iteration effectiveness | Each iteration improves pass rate | Track pass/fail across iterations |

---

## Stretch Goals

1. **Codebase context**: Accept an existing code file as context; generate code that matches the existing style and uses existing utilities
2. **Multi-file generation**: Handle specs that require multiple files (e.g., a class with multiple methods and a separate test file)
3. **Language support**: Extend to JavaScript or TypeScript by swapping the sandbox runner
4. **Spec clarification**: If the spec is ambiguous, ask clarifying questions before generating
5. **Performance benchmarking**: After generating code, benchmark its runtime against naive alternatives

---

## Common Failure Modes

- **Import errors in sandbox**: Generated code imports unavailable packages → add a validation step for imports
- **Infinite recursion**: LLM generates recursive solutions without base cases → add recursion depth check to review
- **Test imports wrong**: Generated tests import the wrong function name → validate test imports match implementation
- **Test gaps**: Generated tests only test happy path → explicitly prompt for edge case coverage
