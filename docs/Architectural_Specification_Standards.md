# Architectural Specification Standards

## 1. Protocol: Autonomous Async Swarm Agents (AASA)

This protocol defines the behavioral and architectural standards for autonomous agents operating within the codebase.

### Core Principles:
*   **Autonomy:** Agents should be capable of independent problem solving within their defined scope.
*   **Asynchrony:** Agent operations should not block the main execution flow where possible. Communication should be event-driven or message-based.
*   **Swarm Intelligence:** Agents should contribute to a shared knowledge base (the "Hive") to enable collective intelligence. In this repo, this is realized through the `KnowledgeBaseService` and `KnowledgeGraphService`.

### Implementation Guidelines:
*   **State Isolation:** Each agent must maintain its own state or use stateless functional patterns to avoid race conditions.
*   **Graceful Failure:** Agents must handle errors internally and report them without crashing the entire system.
*   **Observability:** All agent actions must be logged with high granularity (trace IDs, timestamps) to allow reconstruction of the "swarm's" path.

## 2. Protocol: Additive-Only Development

To maintain system stability and evolution, all changes should strictly adhere to the "Additive-Only" constraint.

### Rules:
*   **No Deletions:** Do not delete existing code unless it is provably dead and has been deprecated for at least one major version cycle.
*   **No Breaking Changes:** Do not modify existing function signatures in a way that breaks backward compatibility. Use optional arguments (`kw_only` is preferred) or create new versions of functions (e.g., `calculate_risk_v2`).
*   **Expansion over Modification:** Instead of changing how `Component A` works, create `Component A_Extended` that inherits from it or wraps it, or add new methods to `Component A`.
*   **Feature Toggling:** New features should be implemented behind flags or as opt-in modules to prevent regression in the core stability.

## 3. Protocol: HNASP (Hierarchical Neural-Symbolic Architecture Standard Protocol)

This protocol governs the integration of Neural (Deep Learning) and Symbolic (Graph/Rule-based) components.

### Standards:
*   **Symbolic Grounding:** Neural outputs (e.g., from LLMs or Transformers) must be grounded in symbolic structures (Knowledge Graph nodes/edges) where possible.
*   **Neural Guidance:** Neural models should guide the traversal or reasoning over symbolic structures, acting as heuristics for search or optimization.
*   **Interface Contract:** The interface between Neural and Symbolic components must be strictly typed (e.g., using Pydantic models) to ensure data integrity during the hand-off.
