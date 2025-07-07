# AGENTS.md - Documentation Guidelines (`docs/`)

This document outlines standards and best practices for creating and maintaining documentation within the `/docs` directory. This includes conceptual documentation, architectural overviews, and detailed explanations of system components or features. This complements code-level documentation (docstrings, comments) and `AGENTS.md` files in other directories.

## 1. Purpose of `/docs` Documentation

*   **Conceptual Understanding:** Provide high-level explanations of the project's goals, key concepts (like the "Probability Map" and "HITL"), and overall architecture.
*   **Design Decisions:** Document significant design choices, their rationale, and any alternatives considered.
*   **User Guidance:** (Future) If the system evolves to have end-users beyond the development team, this directory might host user manuals or guides.
*   **Developer Onboarding:** Help new developers understand the system's structure, components, and workflows.

## 2. Content and Structure

*   **Clarity and Conciseness:** Write clearly and avoid jargon where possible. If technical terms are necessary, define them.
*   **Organization:**
    *   Organize documents logically into subdirectories if needed (e.g., `/docs/architecture`, `/docs/features`).
    *   Use Markdown (`.md`) as the primary format for documentation.
    *   Employ headings, lists, diagrams, and code blocks to structure information and improve readability.
*   **Key Documents to Maintain:**
    *   **Overall System Architecture:** A document describing the major components and how they interact.
    *   **Probability Map Concept:** Detailed explanation of the "AI HITL Risk Control Probability Map" – its goals, data inputs, analytical dimensions, and HITL integration points. (e.g., `Future_State_Probability_Map.md` is a good start).
    *   **Data Flow Diagrams:** Visual representations of how data moves through the system.
    *   **Feature Explanations:** Detailed descriptions of significant features, especially those related to multi-dimensional analysis and HITL capabilities.
    *   **Deployment and Operations Guide:** (Future) If the system is deployed, instructions for deployment, configuration, and monitoring.

## 3. Diagrams and Visuals

*   **Use Diagrams:** Incorporate diagrams (e.g., architecture diagrams, flowcharts, ERDs) to illustrate complex concepts or relationships.
*   **Tools:** Use tools like Mermaid (which can be embedded in Markdown), PlantUML, or dedicated diagramming software. If using external tools, commit the source file (e.g., `.drawio` file) alongside the exported image, or ensure the diagram is easily editable.
*   **Accessibility:** Provide descriptive text alternatives for diagrams if possible.

## 4. Writing Style

*   **Audience Awareness:** Write for the intended audience. For example, documentation for developers can be more technical than documentation for stakeholders.
*   **Consistent Terminology:** Use consistent terminology throughout all documentation, aligning with the definitions in the root `AGENTS.md` and within the codebase.
*   **Active Voice:** Use active voice where possible to make writing more direct and engaging.
*   **Proofread:** Review documentation for grammatical errors, typos, and clarity before committing.

## 5. Documenting New Features and Changes

*   **Contemporaneous Documentation:** Document new features, architectural changes, or significant refactoring *as they are being developed*, not as an afterthought.
*   **Impact Analysis:** When documenting a change, briefly explain its impact on other parts of the system or on existing workflows.
*   **HITL Processes:** Clearly document how Human-in-the-Loop processes are designed to work for any given feature:
    *   What information is presented to the human?
    *   What actions can the human take?
    *   How is human feedback captured and used by the system?

## 6. Review and Maintenance

*   **Peer Review:** Have documentation reviewed by other team members to ensure clarity, accuracy, and completeness.
*   **Keep Documentation Up-to-Date:** Documentation is only useful if it is current. Establish a process for reviewing and updating documentation regularly, especially when related code or system behavior changes. Outdated documentation can be worse than no documentation.
*   **Link to Code:** Where appropriate, link from documentation to relevant sections of the source code or specific `AGENTS.md` files for more detailed, implementation-specific guidance.

By adhering to these guidelines, we can create a comprehensive and valuable set of documentation that supports the development, understanding, and evolution of the AI HITL Risk Control Probability Map.
