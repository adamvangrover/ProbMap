# Adam Sovereign Credit Bundle (v1.0.0)

This package abstracts the logic, schemas, and prompts into a portable, modular format that can be deployed into your repository. It is designed to be ingested by an orchestration framework (like LangChain, AutoGen, or a custom Python runner) to instantiate the "Glass Box" system immediately.

## Directory Structure

```
adam-sovereign-bundle/
├── manifest.yaml                 # Package metadata & model dependencies
├── schemas/                      # The "Data Contracts"
│   ├── sovereign_chunk.json      # Spatial-aware vector schema
│   └── audit_log.json            # Regulatory reporting schema (placeholder)
├── agents/                       # The "Prompt-as-Code" Definitions
│   ├── archivist.yaml            # Retrieval & Citation Logic (placeholder)
│   ├── quant.yaml                # Spreading & Math Logic
│   └── risk_officer.yaml         # Compliance & Critique Logic
├── governance/                   # Executable Audit Rules
│   ├── adr_controls.py           # Python-based architecture tests
│   └── golden_dataset.jsonl      # Test cases for CI/CD (placeholder)
└── infra/                        # Infrastructure-as-Code stubs
    └── vpc_config.tf             # Terraform for the "Iron Bank"
```

## Modules

### Module 1: The Manifest (`manifest.yaml`)
Acts as the configuration entry point, defining the "Sovereign" constraints (e.g., blocking public internet access).

### Module 2: The Agent Persona Library (`agents/`)
YAML-based "Source Code" for agents, using Handlebars-style syntax (`{{variable}}`) for dynamic injection.
- **Quant Agent**: Extracts tabular data and validates accounting identities.
- **Risk Officer Agent**: Critiques the draft credit memo for hallucinations and policy violations.

### Module 3: The Data Contracts (`schemas/`)
JSON schemas enforce the "Polyglot Persistence" strategy, ensuring that data moving between the Vector DB and the Application Layer maintains its spatial context.
- **Sovereign Chunk**: A text chunk with immutable linkage to its source PDF location.

### Module 4: The Compliance Engine (`governance/`)
The "Audit-as-Code" module, designed to be run by your CI/CD pipeline.
- `adr_controls.py`: Contains python functions to audit citation density and financial math.

### Module 5: Infrastructure Stubs (`infra/`)
Reference Terraform configuration for the "Iron Bank" deployment.

## Architecture & Security

This bundle implements a "Glass Box" system architecture.

### The "Iron Bank" Isolation Layer
As defined in `infra/vpc_config.tf` and `manifest.yaml` security constraints:
- The system operates within a strictly private Virtual Private Cloud (VPC).
- The `vector_db_subnet` has `map_public_ip_on_launch` explicitly disabled to prevent public internet access.
- `allow_public_internet` constraint is enforced (`false`).

### Polyglot Persistence & Spatial Context
The `sovereign_chunk.json` data contract ensures data lineage:
- Extracted unstructured data must maintain its linkage to the source document (`doc_id`, `chunk_id`).
- Physical coordinates within original documents are required (`spatial_context.bounding_box`), enabling explicit spatial citations by the `archivist` and `risk_officer` agents rather than relying solely on generated semantic meaning.

## Usage Instruction

To initialize this bundle in your Python environment:

1.  **Clone the structure**: Copy the directory tree into `adam/enterprise_bundle/`.
2.  **Load the Manifest**:
    ```python
    import yaml
    with open("enterprise_bundle/manifest.yaml") as f:
        config = yaml.safe_load(f)
    ```
3.  **Instantiate Agents**: Use a prompt loader to read `agents/quant.yaml` and inject it into your LangChain/AutoGen definition, ensuring the system_prompt is strictly adhered to.

This modular package allows you to version control your logic (Prompts) separately from your application (Python code), satisfying the "Model Risk Management" requirements of modern banking regulators.
