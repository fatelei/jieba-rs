## Description
Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context.

## Type of Change
Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement (non-breaking change that improves performance)
- [ ] Documentation update
- [ ] Code cleanup (non-breaking change that improves code quality)
- [ ] Other (please describe):

## Testing
Please describe the tests you ran to verify your changes:

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks (if applicable)
- [ ] Manual testing (please describe)

### Test Results
```bash
# Example test output
cargo test
# Running unittests (target/debug/deps/rust_jieba-...)
# test test_chinese_segmentation ... ok
# test test_empty_string ... ok
# test result: ok. 2 passed; 0 failed; 0 ignored
```

## Benchmark Results (if applicable)
If your changes affect performance, please provide benchmark results:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Speed  | 0.001s | 0.0008s | 20% |
| Memory | 10MB   | 8MB    | 20% |

## Code Quality
Please confirm the following:

- [ ] I have followed the existing code style
- [ ] I have added comments to new or complex code
- [ ] I have updated the documentation if necessary
- [ ] My code follows the project's contribution guidelines

## Breaking Changes
If this PR introduces breaking changes, please describe them:

1. **Change Description**: What changed
2. **Impact**: Who is affected
3. **Migration**: How to update existing code

## Additional Notes
Any additional information or context that reviewers should be aware of.

## Related Issues
Fixes #123
Closes #456