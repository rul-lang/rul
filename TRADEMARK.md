# Rul™ Trademark & Specification Conformance Policy

The **Rul** source code and specification documents in this repository are dual-licensed under the **MIT OR Apache-2.0** open-source licenses (see [`LICENSE-MIT`](./LICENSE-MIT), [`LICENSE-APACHE`](./LICENSE-APACHE), and [`NOTICE`](./NOTICE)).

However, following the industry-standard governance model of the **Rust Foundation** and the **Linux Foundation**, **open-source copyright licenses do not grant an unrestricted license to use the Rul trademarks** (specifically covered by Section 6 of the Apache License, Version 2.0).

---

## 1. Protected Marks & Identifiers

The following names, identifiers, and marks (collectively, the **"Rul Marks"**) are protected trademarks and provenance identifiers of the Original Creator (**Tran Trong Tri / Tymon Tran**):

1. The wordmarks **"Rul"**, **"Rul Language"**, **"rul-lang"**, and **"Rul Engine"**.
2. The official **"Rul Certified"** / **"Rul Conformant (.rl)"** compatibility designations.
3. The canonical binary magic header **`RL_MAGIC` (`b"RUL\x00\x01\x00\x00\x00"`)** when used to claim conformance with the official Rul Binary Execution Specification ([`docs/rul_spec_v0.md`](./docs/rul_spec_v0.md)).

---

## 2. Permitted Uses (No Prior Permission Required)

You are encouraged and permitted to use the Rul Marks without prior written permission in the following ways:

1. **Unmodified Redistribution**: Compiling and distributing unmodified binaries built directly from the canonical repository (`https://github.com/rul-lang/rul`).
2. **Ecosystem Tooling & Integration**: Naming libraries, MCP servers, adapters, or tools that interoperate with Rul using clear descriptive phrasing such as *"MCP Server for Rul"*, *"Python Bindings for Rul (.rl)"*, or *"Powered by Rul"*, provided you do not imply official endorsement where none exists.
3. **Academic & Benchmark Publication**: Referencing **Rul (`.rl`)** in research papers, benchmarks, blog posts, and talks, provided attribution to the original author and repository is preserved per [`CITATION.cff`](./CITATION.cff) and [`NOTICE`](./NOTICE).

---

## 3. Prohibited Uses (Protecting Specification Integrity & Preventing Brand Hijacking)

To prevent fragmentation, security downgrade attacks on AI Agents, and confusion in the ecosystem, you **MUST NOT**:

1. **Distribute Incompatible Forks as "Rul"**:
   - If you modify the `.rl` binary format ([`src/bytecode.rs`](./src/bytecode.rs)), alter opcode semantics, or weaken/bypass any check in the $O(N)$ static verifier ([`src/verifier.rs`](./src/verifier.rs) — including topological SSA ordering, `AffineToken` / `ReadLease` linear ownership, `ProofEntry` bounds verification, or `CapabilityMask` sandboxing), you **must not** distribute the resulting software under the name **`Rul`**, **`rul`**, or **`rul-bench`**, and you **must** change the 8-byte header magic (`RL_MAGIC`) to a distinct identifier.
2. **Remove Attribution Notices**:
   - Remove or obscure the copyright and creator attribution notices in [`LICENSE-MIT`](./LICENSE-MIT), [`LICENSE-APACHE`](./LICENSE-APACHE), or [`NOTICE`](./NOTICE).
3. **Register Confusingly Similar Domains or Organizations**:
   - Register domain names, cloud service names, or package namespaces that imply you are the official source or creator of the Rul programming language.

---

## 4. Contact & Foundation Governance

As Rul transitions into a neutral open-source foundation (such as the **Linux Foundation / LF AI & Data**), stewardship of the Rul Marks will be governed in accordance with foundation trademark policies.

For trademark inquiries or official conformance certification:
- **Creator & Lead Architect**: Tran Trong Tri (Tymon Tran) — `tymon3568@gmail.com`
- **Repository**: `https://github.com/rul-lang/rul`
