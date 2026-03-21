# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of TrustformeRS seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please do NOT:
- Open a public issue
- Post about it on social media
- Share details publicly before a fix is available

### Please DO:
- Email us at: security@trustformers.rs
- Encrypt your message using our PGP key (available at [link])
- Allow us reasonable time to respond before disclosure

### What to include:
1. Type of vulnerability (e.g., buffer overflow, SQL injection, cross-site scripting)
2. Full paths of source file(s) related to the vulnerability
3. Location of the affected source code (tag/branch/commit or direct URL)
4. Step-by-step instructions to reproduce the issue
5. Proof-of-concept or exploit code (if possible)
6. Impact of the issue, including how an attacker might exploit it

### What to expect:
- **Initial Response**: Within 48 hours, confirming receipt
- **Assessment**: Within 7 days, we'll assess the vulnerability and its severity
- **Fix Timeline**: We'll provide an estimated timeline for a fix
- **Disclosure**: We'll coordinate disclosure timing with you

## Security Best Practices

When using TrustformeRS:

### Model Security
- Always validate model inputs
- Sanitize outputs before using in sensitive contexts
- Be cautious with models from untrusted sources
- Verify model checksums when downloading

### Code Security
```rust
// Good: Input validation
let input = sanitize_input(user_input)?;
let output = model.forward(&input)?;

// Bad: Direct user input
let output = model.forward(&user_input)?; // Potential injection
```

### Deployment Security
- Use latest stable version with security patches
- Enable secure communication (TLS) for model serving
- Implement rate limiting for inference endpoints
- Monitor for unusual usage patterns

### Data Security
- Never log sensitive data
- Implement proper access controls
- Encrypt model weights if they contain sensitive information
- Clear GPU memory after processing sensitive data

## Known Security Considerations

### Memory Safety
TrustformeRS is written in Rust, providing memory safety guarantees. However:
- Unsafe code blocks are audited regularly
- FFI boundaries (CUDA, etc.) require extra caution
- Custom kernels should be thoroughly tested

### Model Attacks
Be aware of these potential attacks:
- **Adversarial Examples**: Inputs designed to fool models
- **Model Extraction**: Attempts to steal model weights
- **Data Poisoning**: Malicious training data
- **Membership Inference**: Determining if data was in training set

### Mitigation Strategies
```rust
// Input validation
pub fn validate_input(tensor: &Tensor) -> Result<()> {
    ensure!(tensor.is_finite(), "Input contains non-finite values");
    ensure!(tensor.abs().max() < 1000.0, "Input values out of range");
    Ok(())
}

// Output sanitization  
pub fn sanitize_output(output: &str) -> String {
    // Remove potential XSS/injection attempts
    html_escape::encode_safe(output).to_string()
}
```

## Security Updates

Security updates will be released as:
- **Critical**: Immediate patch release (x.y.z+1)
- **High**: Within 7 days
- **Medium**: Within 30 days
- **Low**: Next regular release

Subscribe to security announcements:
- GitHub Security Advisories
- Mailing list: security-announce@trustformers.rs
- RSS feed: https://trustformers.rs/security/feed.xml

## Acknowledgments

We appreciate security researchers who help keep TrustformeRS secure. Contributors will be acknowledged (unless they prefer to remain anonymous) in:
- Security advisories
- Release notes
- Our security hall of fame

## Contact

- **Security issues**: security@trustformers.rs
- **PGP Key**: [0xDEADBEEF](https://trustformers.rs/pgp-key.asc)
- **General questions**: GitHub Discussions

Thank you for helping keep TrustformeRS and its users safe!