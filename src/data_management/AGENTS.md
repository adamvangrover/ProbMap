# AGENTS.md - Data Management (`src/data_management/`)

This guide provides specific instructions for developing and maintaining components within the `src/data_management/` directory, including the `Ontology`, `KnowledgeBaseService`, and `KnowledgeGraphService`. It builds upon guidelines in `src/AGENTS.md` and the root `AGENTS.md`.

## 1. Ontology (`ontology.py`)

*   **Pydantic Models as Single Source of Truth:** The Pydantic models in `ontology.py` define the schema and validation rules for all core data entities (e.g., `CorporateEntity`, `LoanAgreement`, `FinancialStatement`). These models are the canonical representation of our data structures.
*   **Evolution and Versioning:**
    *   Changes to the ontology (e.g., adding fields, changing types) MUST be carefully considered for their impact on downstream consumers (models, services, API).
    *   For significant changes, consider versioning strategies or ensure backward compatibility where possible. Discuss such changes with the team.
    *   New fields should generally be `Optional` to avoid breaking existing data loading unless a default value is always appropriate.
*   **Clarity and Documentation:**
    *   All Pydantic model fields MUST have clear, descriptive names.
    *   Include docstrings for each model and for complex fields explaining their meaning and purpose.
    *   Use `Field` from Pydantic for descriptions, examples, and validation rules (e.g., `gt`, `le`).
*   **Enumerations:** Use Python `Enum` (or `StrEnum`) for fields with a fixed set of predefined values (e.g., `IndustrySector`, `Currency`, `CollateralType`).

## 2. Knowledge Base (`knowledge_base.py`)

*   **Purpose:** The `KnowledgeBaseService` is responsible for ingesting, validating (against the Ontology), storing, and providing access to the foundational data of the system.
*   **Data Ingestion:**
    *   When adding loaders for new data sources (e.g., new files, database tables, APIs):
        *   Implement robust parsing and error handling for the specific data format.
        *   Ensure all ingested data is validated against the corresponding Pydantic models from the Ontology. Log validation errors clearly.
        *   Handle missing or malformed data gracefully (e.g., skip record with warning, impute with care if appropriate and documented).
*   **HITL Data Integration:**
    *   The `KnowledgeBaseService` will be responsible for storing and retrieving HITL annotations.
    *   Design clear interfaces (methods) for:
        *   Storing validated qualitative scores (e.g., overridden management quality).
        *   Storing analyst feedback on model predictions (e.g., flags, reason codes, notes).
        *   Storing custom tags or classifications applied to entities.
    *   The storage mechanism for these annotations should be decided (e.g., separate JSON files, new tables in a DB if we move away from files) and encapsulated within this service.
*   **Querying and Access:**
    *   Methods providing access to data should return objects validated by the Ontology (e.g., a list of `LoanAgreement` objects).
    *   Optimize data retrieval for common access patterns.
*   **Extensibility for New Data Environments:**
    *   Future work will involve abstracting the data storage backend. Current file-based loading is one implementation. Design with the intent to support other backends (databases, APIs) by defining clear internal interfaces for data operations (load, store, query).

## 3. Knowledge Graph (`knowledge_graph.py`)

*   **Purpose:** The `KnowledgeGraphService` constructs and provides query capabilities for a graph representation of entities and their relationships.
*   **Graph Population:**
    *   Ensure that all nodes and edges added to the graph are consistent with the Ontology and data from the `KnowledgeBaseService`.
    *   Clearly define `RelationshipType` enums for all semantic relationships in the graph.
    *   Handle cases where related entities might not exist (e.g., a subsidiary ID mentioned but no full company profile for it) by creating placeholder nodes or logging warnings.
*   **Node and Edge Attributes:** Store relevant attributes on nodes and edges, sourced from the `KnowledgeBaseService` or computed.
*   **Query Capabilities:**
    *   Develop new graph algorithms or query methods based on analytical requirements (e.g., identifying specific risk patterns, contagion paths, influential entities).
    *   Ensure query methods are efficient, especially for large graphs.
*   **HITL Curation of Graph:**
    *   (Future) If graph relationships are inferred or if anomalies are detected, this service might need methods to incorporate human validation or correction of graph structures.
*   **Synchronization:** The KG is built from the KB. Consider strategies for keeping the KG synchronized if the underlying KB data changes frequently (currently, it's rebuilt on initialization).

## 4. General Considerations for Data Management

*   **Data Quality:** Emphasize data quality checks at all stages: ingestion, validation, and before use in models or services.
*   **Immutability:** Treat data loaded from source files as immutable within a session where possible. Transformations should create new data structures rather than modifying loaded raw data in place, unless explicitly intended (e.g. for an ETL process).
*   **Logging:** Log significant data loading events, validation errors, and statistics about the loaded data (e.g., number of companies, loans).

Adherence to these guidelines will ensure that the data management layer is robust, reliable, and can support the evolving needs of the AI HITL Risk Control Probability Map.
