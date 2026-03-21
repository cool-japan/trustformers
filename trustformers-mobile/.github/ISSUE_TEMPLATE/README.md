# Issue Templates Guide

This directory contains GitHub issue templates for the TrustformeRS project. These templates help standardize issue reporting and ensure we collect all necessary information to address issues efficiently.

## Available Templates

### 1. 🐛 Bug Report (`bug_report.yml`)
Use this template when you encounter unexpected behavior, crashes, or errors in TrustformeRS.

**When to use:**
- Runtime errors or panics
- Incorrect computation results
- Memory leaks or performance regressions
- Build or compilation failures

**Key information collected:**
- Detailed reproduction steps
- Error messages and stack traces
- System configuration
- Minimal reproducible example

### 2. 🚀 Feature Request (`feature_request.yml`)
Use this template to propose new features or enhancements to TrustformeRS.

**When to use:**
- Suggesting new model architectures
- Proposing API improvements
- Requesting new optimization techniques
- Adding platform support

**Key information collected:**
- Problem statement and use cases
- Proposed solution and API design
- Alternative approaches considered
- Potential impact on existing users

### 3. 📚 Documentation Improvement (`documentation.yml`)
Use this template to report documentation issues or suggest improvements.

**When to use:**
- Fixing typos or grammatical errors
- Clarifying confusing explanations
- Adding missing examples
- Updating outdated information

**Key information collected:**
- Location of documentation issue
- Suggested improvements
- Target audience consideration

### 4. ⚡ Performance Issue (`performance_issue.yml`)
Use this template specifically for performance-related problems.

**When to use:**
- Slower than expected execution
- High memory consumption
- Low GPU/CPU utilization
- Performance regressions

**Key information collected:**
- Performance metrics and comparisons
- Hardware specifications
- Profiling data
- Reproduction code

### 5. 🤖 Model Request (`model_request.yml`)
Use this template to request implementation of new model architectures.

**When to use:**
- Requesting state-of-the-art models
- Proposing domain-specific architectures
- Suggesting mobile-optimized models

**Key information collected:**
- Model architecture details
- Research papers and references
- Use cases and benchmarks
- Implementation complexity

### 6. ❓ Question / Help (`question.yml`)
Use this template when you need help or have questions about using TrustformeRS.

**When to use:**
- Implementation guidance
- Best practices questions
- Debugging assistance
- Architecture decisions

**Key information collected:**
- Context and goals
- What you've already tried
- Related code snippets
- Urgency level

## Template Features

### Form-based Templates
All templates use GitHub's form-based issue templates (YAML format) which provide:
- Structured data collection
- Required vs optional fields
- Input validation
- Better mobile experience
- Consistent formatting

### Smart Defaults
Templates include:
- Pre-filled labels
- Suggested titles with prefixes
- Dropdown menus for common options
- Placeholder text with examples

### Progressive Disclosure
Templates are designed to:
- Start with essential information
- Add detail progressively
- Include optional advanced sections
- Provide examples throughout

## For Maintainers

### Processing Issues

1. **Triage Labels**: All issues start with `needs-triage` label
2. **Auto-assignment**: Use CODEOWNERS for automatic assignment
3. **Priority Labels**: Add priority labels based on impact
4. **Component Labels**: Add component-specific labels

### Label Structure

```
Type Labels:
- bug
- enhancement
- documentation
- performance
- model-request
- question

Priority Labels:
- priority-critical
- priority-high
- priority-medium
- priority-low

Status Labels:
- needs-triage
- needs-reproduction
- needs-more-info
- in-progress
- blocked

Component Labels:
- component-core
- component-models
- component-optim
- component-mobile
- component-serving
```

### Issue Workflow

1. **New Issue** → `needs-triage`
2. **Triaged** → Remove `needs-triage`, add priority/component
3. **In Progress** → Add `in-progress`, assign developer
4. **Blocked** → Add `blocked`, document blockers
5. **Resolved** → Close with resolution comment

## For Contributors

### Before Opening an Issue

1. **Search existing issues** - Your issue may already be reported
2. **Check documentation** - The answer might be in our docs
3. **Prepare reproduction** - Have a minimal example ready
4. **Gather information** - System specs, versions, error messages

### Choosing the Right Template

Use this decision tree:
```
Is something broken? → Bug Report
Want something new? → Feature Request
Model missing? → Model Request  
Docs unclear? → Documentation
Slow performance? → Performance Issue
Need help? → Question
```

### Issue Best Practices

1. **One issue per problem** - Don't combine multiple issues
2. **Clear titles** - Be specific about the problem
3. **Complete information** - Fill all required fields
4. **Code formatting** - Use code blocks with syntax highlighting
5. **Be respectful** - Follow our Code of Conduct

### After Opening an Issue

- **Monitor notifications** - Respond to maintainer questions
- **Provide updates** - If you find new information
- **Test fixes** - Help verify proposed solutions
- **Close if resolved** - Let us know if you solve it yourself

## Customizing Templates

### Adding New Templates

1. Create new YAML file in `.github/ISSUE_TEMPLATE/`
2. Follow the existing template structure
3. Add to this README
4. Update `config.yml` if needed

### Template Structure

```yaml
name: Template Name
description: Brief description
title: "[PREFIX] "
labels: ["label1", "label2"]
body:
  - type: markdown
    attributes:
      value: Introduction text
  
  - type: textarea
    id: field-id
    attributes:
      label: Field Label
      description: Help text
      placeholder: Example text
    validations:
      required: true
```

### Available Field Types

- `markdown` - Static text/instructions
- `textarea` - Multi-line text input  
- `input` - Single-line text input
- `dropdown` - Single selection
- `checkboxes` - Multiple selections

## Integration with Other Tools

### GitHub Actions
Templates can trigger workflows:
```yaml
on:
  issues:
    types: [opened, labeled]
```

### Project Boards
Auto-add issues to projects based on labels

### Webhooks
Send notifications to Slack/Discord for new issues

## External Links

See `config.yml` for external links:
- Discord community
- Documentation
- Discussions
- Security policy

## Maintenance

### Regular Reviews
- Review template effectiveness quarterly
- Update based on common patterns
- Archive outdated templates
- Gather contributor feedback

### Metrics to Track
- Template usage rates
- Issue completion rates
- Time to resolution
- User satisfaction

## Contributing to Templates

To improve our templates:
1. Open a PR with your changes
2. Explain the motivation
3. Show examples of issues that would benefit
4. Get review from maintainers

Remember: Good templates lead to good issues, which lead to faster resolutions!