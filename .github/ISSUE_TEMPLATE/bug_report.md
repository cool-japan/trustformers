---
name: Bug Report
about: Create a report to help us improve TrustformeRS
title: "[BUG] "
labels: ["bug", "needs-triage"]
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Reproduction Steps
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Environment Information
- **TrustformeRS Version**: [e.g. 0.1.0]
- **Rust Version**: [e.g. 1.75.0]
- **Python Version** (if using Python bindings): [e.g. 3.11.0]
- **Operating System**: [e.g. macOS 14.0, Ubuntu 22.04, Windows 11]
- **Hardware**: [e.g. M1 Mac, x86_64, ARM64]
- **GPU**: [e.g. NVIDIA RTX 4090, Apple M1, AMD RX 7900]
- **Backend**: [e.g. CPU, CUDA, ROCm, Metal, OpenVINO]

## Code Sample
If applicable, add a minimal code sample to reproduce the issue:

```rust
// Your code here
```

or

```python
# Your Python code here
```

## Error Messages/Logs
If applicable, add error messages or log outputs:

```
Paste error messages here
```

## Additional Context
Add any other context about the problem here, including:
- Screenshots or videos if applicable
- Related issues or discussions
- Workarounds you've tried
- Performance impact

## Component
Which TrustformeRS component is affected? (check all that apply)
- [ ] trustformers-core (tensor operations, hardware acceleration)
- [ ] trustformers-models (model implementations)
- [ ] trustformers-training (training infrastructure)
- [ ] trustformers-optim (optimizers)
- [ ] trustformers-tokenizers (tokenization)
- [ ] trustformers-serve (serving infrastructure)
- [ ] trustformers-py (Python bindings)
- [ ] trustformers-js (JavaScript/WASM bindings)
- [ ] trustformers-c (C API)
- [ ] trustformers-mobile (mobile deployment)
- [ ] trustformers-wasm (WebAssembly)
- [ ] trustformers-debug (debugging tools)
- [ ] Documentation
- [ ] CI/CD
- [ ] Other: _____________

## Priority
How critical is this issue?
- [ ] Critical (blocks development/production)
- [ ] High (significant impact on functionality)
- [ ] Medium (moderate impact)
- [ ] Low (minor issue or enhancement)

## Checklist
- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided all relevant environment information
- [ ] I have included a minimal reproduction case
- [ ] I have checked the documentation