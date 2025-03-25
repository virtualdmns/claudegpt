# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within ClaudeGPT, please send an email to virtualdmns@gmail.com. All security vulnerabilities will be promptly addressed.

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

## Security Best Practices

When using ClaudeGPT, please follow these security best practices:

1. **API Keys**: Never hardcode API keys in your code. Always use environment variables or a secure secrets management system.

2. **Environment Variables**: Use the `.env` file for local development, but ensure it's listed in `.gitignore` to prevent accidental commits.

3. **Input Validation**: When extending ClaudeGPT with custom tools, always validate and sanitize inputs.

4. **Dependency Management**: Regularly update dependencies to ensure you have the latest security patches.

5. **Access Control**: Implement appropriate access controls when deploying applications built with ClaudeGPT.

## Disclosure Policy

When we receive a security bug report, we will:

- Confirm the problem and determine the affected versions
- Audit code to find any potential similar problems
- Prepare fixes for all supported versions
- Release new versions as soon as possible