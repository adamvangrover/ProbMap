# AGENTS.md - Top-Level Guidance for Project Development

## 1. Project Vision & Core Objective

This project aims to develop an **AI Human-in-the-Loop (HITL) Risk Control Probability Map**. The system should provide a dynamic, multi-dimensional view of corporate credit risk, integrating quantitative models, qualitative data, knowledge graph context, and scenario analysis to output actionable insights.

The "Probability Map" itself is a conceptual framework for visualizing and analyzing risk across various dimensions and data environments. HITL capabilities are crucial for leveraging expert judgment, validating AI outputs, and continuously improving the system.

## 2. Core Architectural Principles

*   **Modularity:** Design components (data management, risk models, services, API) with clear responsibilities and well-defined interfaces to promote independent development, testing, and maintenance.
*   **Testability:** All new functionalities should be accompanied by appropriate unit and integration tests. Strive for high test coverage.
*   **Data-Driven Decisions:** System enhancements and model improvements should be guided by data analysis and empirical evidence.
*   **Separation of Concerns:** Maintain a clear distinction between data persistence, business logic, modeling, and presentation layers.
*   **Extensibility:** Build components with future enhancements in mind. For example, data services should be adaptable to new data sources, and modeling frameworks should allow for the integration of new model types.

## 3. Human-in-the-Loop (HITL) Philosophy

*   **Humans Augment AI, AI Supports Humans:** HITL is not just about correcting AI; it's about creating a synergistic relationship where human expertise guides and refines AI, and AI provides tools and insights to enhance human decision-making.
*   **Feedback Loops are Critical:** Design HITL interactions to provide structured feedback that can be used to improve models, data quality, and processes over time.
*   **Transparency and Explainability:** Strive to make AI components (especially models) as transparent and explainable as possible to facilitate effective human review and build trust.
*   **User-Centric Design:** HITL interfaces and workflows should be designed with the analyst's needs and tasks in mind.

## 4. General Development Conventions

*   **Coding Standards:**
    *   All Python code MUST adhere to PEP 8 style guidelines. Use linters (e.g., Flake8, Pylint) and formatters (e.g., Black, Ruff) to ensure consistency.
    *   Follow established best practices for object-oriented design and functional programming where appropriate.
*   **Logging:** Implement comprehensive logging across all modules. Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL). Logs should be informative and help in debugging and monitoring. Refer to `src/AGENTS.md` for more specific logging guidelines.
*   **Error Handling:** Implement robust error handling. Use specific exception types where possible. Avoid catching generic `Exception` unless necessary, and always log caught exceptions.
*   **Version Control:**
    *   Follow a consistent branching strategy (e.g., Gitflow-like feature branches).
    *   Write clear and descriptive commit messages. The subject line should be concise (<=50 chars), followed by a blank line and a more detailed body if necessary.
*   **Dependency Management:** Use `requirements.txt` for managing Python dependencies. Keep it updated and minimize unnecessary dependencies.

## 5. Documentation Standards

*   **Code Comments:** Write clear and concise comments to explain complex logic or non-obvious decisions.
*   **Docstrings:** All public modules, classes, functions, and methods MUST have docstrings (e.g., following Google Python Style Guide or NumPy/SciPy conventions).
*   **External Documentation:** Significant architectural decisions, new service designs, and complex workflows should be documented (e.g., in Markdown files within the `/docs` directory or relevant module directories). Refer to `docs/AGENTS.md` for more details.
*   **`AGENTS.md` Files:** These files provide specific instructions for agents (like yourself) working on different parts of the codebase. Always check the relevant `AGENTS.md` files before modifying code in a directory.

## 6. Key Terminology

*   **Probability Map:** The overarching system/framework for multi-dimensional risk analysis and visualization.
*   **Risk Item:** A structured data object representing a single unit of risk (e.g., a loan, a company) with its associated metrics and attributes.
*   **HITL Annotation:** Data or feedback provided by a human expert (e.g., a validated score, a review flag, a textual comment).
*   **Dimensional Analysis:** The ability to filter, group, and aggregate risk data across various attributes (e.g., sector, country, risk score, custom tags).

This document provides high-level guidance. More specific instructions can be found in `AGENTS.md` files within relevant subdirectories. If any instruction here conflicts with a user request for this specific task, the user request takes precedence.
