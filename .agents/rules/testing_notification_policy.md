# Testing & Command Execution Notification Policy

## Core Invariant
- **Never run tests or commands without explicit prior notification**:
  Before running any script, unit test, simulation benchmark, or execution command, you MUST explicitly inform the user in simple terms:
  1. The exact command and file being run.
  2. The purpose of the test (what behavior is being verified).
  3. The key parameters involved (e.g. number of workers, grid size, resolution).

## Language & Direction
- All pre-execution notifications and chat responses must be written in Hebrew wrapped in `<div dir="rtl">`.
