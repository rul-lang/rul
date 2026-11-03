# Security Policy

## Supported Versions

Security updates and verifier soundness fixes are applied to the latest minor release on the `main` branch:

| Version | Supported |
| :--- | :--- |
| `0.4.x` (`main`) | :white_check_mark: Yes |
| `< 0.4.0` | :x: Upgrade to `>= 0.4.0` |

---

## Scope of Security & Soundness Vulnerabilities

Because **Rul (`.rl`)** is designed as a zero-trust, proof-carrying execution engine and native AOT compiler for Autonomous AI Agents, we treat the following classes of bugs as **security vulnerabilities**:

1. **$O(N)$ Static Verifier Soundness Bypass (`src/verifier.rs`)**:
   - Any malformed `.rl` binary or `.rls` program that passes `verify_program` while violating SSA topological dominance, `AffineConsume`/`ReadLease` linear ownership, `ProofEntry` slice/gather bounds, or `CapabilityMask` hardware sandboxing.
2. **AOT Native Machine Code Safety & Equivalence Escape (`src/aot.rs`)**:
   - Any verified `.rl` program that compiles via `compile_to_elf64_bytes` into an `x86_64` `ELF64` binary that performs out-of-bounds memory access, executes uninitialized stack slots, bypasses `AssertContract` / `Gather` bounds checks, or silently diverges from the Tier 0 Golden Oracle.
3. **Denial of Service (`DoS`) or Panic in Zero-Panic Entrypoints (`src/lib.rs`, `src/ffi.rs`, `src/mcp.rs`)**:
   - Any `.rl` byte payload or JSON-RPC input that triggers a panic, integer overflow crash, or algorithmic complexity blowup ($> O(N)$) in `RulProgram::from_bytes`, `verify_program`, `rul-ffi`, or `rul-mcp`.

---

## Reporting a Vulnerability

**Please do NOT report security or verifier soundness vulnerabilities through public GitHub Issues or Discussions.**

Instead, please report them privately using one of the following channels:

1. **GitHub Private Vulnerability Reporting (Preferred)**:
   - Submit a private advisory directly at: **[https://github.com/rul-lang/rul/security/advisories/new](https://github.com/rul-lang/rul/security/advisories/new)**
2. **Direct Maintainer Email**:
   - Send an email to **Tran Trong Tri (Tymon Tran)** at `<tymon3568@gmail.com>` with the subject line `[RUL SECURITY] <brief summary>`.

### What to Include in Your Report

- The exact `.rls` symbolic listing or hex-encoded `.rl` binary payload that reproduces the issue.
- The command or API entrypoint invoked (`rul check`, `rul run`, `verify_program`, `compile_to_elf64_bytes`, etc.).
- Expected behavior vs. actual behavior (e.g., expected `VerifyErrorKind` vs. actual verifier pass or native crash).

### Response Timeline

- **Initial Acknowledgment**: Within **48 hours**.
- **Triage & Fix Verification**: Within **7 days** for critical verifier or AOT soundness issues, followed by a coordinated release and credit in the security advisory.
