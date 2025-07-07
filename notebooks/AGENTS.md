# AGENTS.md - Jupyter Notebooks (`notebooks/`)

This document provides guidelines for creating and maintaining Jupyter notebooks within the `notebooks/` directory. These notebooks serve various purposes, including exploratory data analysis (EDA), model prototyping, system demonstration, and generating analytical reports.

## 1. Purpose and Audience

*   **Clarity of Purpose:** Each notebook should have a clear, stated purpose at the beginning (e.g., in a Markdown cell).
*   **Target Audience:** Consider the intended audience (e.g., data scientists, analysts, developers, stakeholders) and tailor the language, level of detail, and visualizations accordingly.

## 2. Notebook Structure and Organization

*   **Standard Sections:** Organize notebooks into logical sections using Markdown headings. Common sections include:
    1.  **Title and Introduction:** Clearly state the notebook's purpose and provide a brief overview.
    2.  **Setup/Imports:** Import all necessary libraries and configure settings (e.g., logging, plotting styles).
    3.  **Data Loading:** Load any required data, clearly indicating sources.
    4.  **Analysis/Methodology:** The main body of the notebook, detailing steps, code, and interpretations.
    5.  **Visualizations:** Present data and results graphically.
    6.  **Results/Findings:** Summarize key results.
    7.  **Conclusion/Next Steps:** Conclude the analysis and suggest potential future work or actions.
*   **Cell Management:**
    *   Keep code cells relatively short and focused on a specific task.
    *   Use Markdown cells extensively to explain the code, interpret results, and provide narrative flow.
    *   Clear cell outputs before committing, unless the output is essential for understanding the notebook's state (e.g., a specific plot or table that is frequently referenced). For large outputs, consider saving them to a file and loading them instead of embedding them directly.

## 3. Code Quality and Reproducibility

*   **Readability:** Write clean, well-commented Python code, adhering to PEP 8 guidelines.
*   **Reproducibility:**
    *   Ensure that notebooks can be run from top to bottom without errors by someone else (or your future self) in a clean environment with the project's dependencies installed.
    *   Set random seeds if stochastic processes are involved to ensure consistent outputs.
    *   Clearly specify any required environment setup or data dependencies.
*   **Avoid Hardcoding Paths:** Use relative paths or paths derived from `src.core.config.settings` where possible, rather than absolute paths.
*   **Modular Code:** For complex or reusable logic, consider moving it into Python scripts within the `src/` directory and importing it into the notebook. This keeps notebooks cleaner and promotes code reuse.

## 4. Visualizations

*   **Clarity and Purpose:** Each visualization should have a clear purpose in conveying information.
*   **Appropriate Chart Types:** Choose chart types that are appropriate for the data and the insight you want to communicate.
*   **Labels and Titles:** All plots MUST have clear titles, axis labels (with units where applicable), and legends if multiple series are shown.
*   **Aesthetics:** Use consistent and visually appealing styles. Consider using libraries like Seaborn or Matplotlib styles.
*   **Saving Plots:** If plots are important outputs, save them to files (e.g., in the `output/` directory) with descriptive names, in addition to displaying them in the notebook.

## 5. Demonstrating System Capabilities

*   **Service Interaction:** When demonstrating system capabilities (e.g., in `02_comprehensive_risk_analysis.ipynb`), clearly show how different services (`KnowledgeBaseService`, `PDModel`, `RiskMapService`, etc.) are instantiated and used.
*   **Illustrating HITL:**
    *   Notebooks can be used to simulate or demonstrate HITL workflows. For example:
        *   Show a raw model output.
        *   Provide a Markdown cell explaining a hypothetical analyst's review.
        *   Show how a (simulated) human override or annotation would be applied.
        *   Demonstrate how this HITL input could affect subsequent calculations or visualizations.
*   **"Probability Map" Visualization:** Use notebooks as the primary means to prototype and display visualizations related to the "Probability Map" concept, especially before a dedicated UI is available.

## 6. Maintenance

*   **Keep Notebooks Updated:** As the underlying `src/` code evolves, ensure that notebooks using that code are updated accordingly to prevent them from becoming outdated or broken.
*   **Version Control:** Commit notebooks to version control. Be mindful of large outputs in committed notebooks; use tools like `nbstripout` if necessary to clear output before committing.

By following these guidelines, our Jupyter notebooks will serve as valuable assets for analysis, demonstration, and communication throughout the project.
