---
name: Bug Report
about: Create a report to help us improve
title: "[BUG] "
labels: bug
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Environment
Please complete the following information:
- OS: [e.g. Linux, macOS, Windows]
- Python Version: [e.g. 3.8, 3.9, 3.10]
- Rust Version: [e.g. 1.70, 1.75]
- Rust jieba Version: [e.g. 0.1.0, 0.2.0]

## Input Text
Please provide the exact Chinese text that caused the issue:

```text
你的中文文本
```

## Expected Output
```text
期望的分词结果
```

## Actual Output
```text
实际的分词结果
```

## Code Example
```python
import rust_jieba

# Your code that reproduces the bug
text = "你的中文文本"
result = rust_jieba.cut(text)
print(list(result))
```

## Additional Context
Add any other context about the problem here.

## Possible Solution
If you have any ideas on how to fix this bug, please describe them here.

## Checklist
- [ ] I have searched the existing issues for similar bug reports
- [ ] I have provided a minimal, reproducible example
- [ ] I have included all relevant environment information
- [ ] I have checked that this is not a duplicate issue