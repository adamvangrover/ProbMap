import csv
import json
import random
from datetime import datetime, timedelta

# Configuration
NUM_NEW_COMPANIES = 12
NUM_NEW_LOANS = 25  # Approximate, may vary slightly
YEARS_FINANCIAL_STATEMENTS = [2022, 2023] # Generate for 2-3 years

# --- Data Generation Functions ---

def generate_id(prefix, current_max_id):
    """Generates a new ID based on the prefix and current max numeric part."""
    if isinstance(current_max_id, str):
        numeric_part = int(current_max_id.replace(prefix, ""))
    else: # assuming it's already an int
        numeric_part = current_max_id
    return f"{prefix}{numeric_part + 1:03d}" # For COMP, FS, DE - 3 digits

def generate_loan_id(prefix, current_max_id):
    """Generates a new Loan ID based on the prefix and current max numeric part."""
    if isinstance(current_max_id, str):
        numeric_part = int(current_max_id.replace(prefix, ""))
    else: # assuming it's already an int
        numeric_part = current_max_id
    return f"{prefix}{numeric_part + 1:04d}" # For LOAN - 4 digits


def get_max_existing_id(existing_data, id_field, prefix):
    """Finds the maximum numeric part of existing IDs."""
    max_id_num = 0
    if not existing_data: # If starting from scratch or file was empty
        if prefix == "COMP": return 8
        if prefix == "LOAN": return 7007
        if prefix == "FS": return 11
        if prefix == "DE": return 2
        return 0

    for item in existing_data:
        item_id = item.get(id_field, "")
        if item_id.startswith(prefix):
            try:
                num = int(item_id.replace(prefix, ""))
                if num > max_id_num:
                    max_id_num = num
            except ValueError:
                continue
    return max_id_num

def generate_new_companies(num_companies, max_existing_comp_id_num):
    """Generates new company data."""
    new_companies = []
    current_max_id_num = max_existing_comp_id_num

    for i in range(num_companies):
        current_max_id_num += 1
        company_id = f"COMP{current_max_id_num:03d}"
        new_companies.append({
            "company_id": company_id,
            "company_name": f"New Company {chr(65 + i)}{current_max_id_num}",
            "industry": random.choice(["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"]),
            "country": random.choice(["USA", "Canada", "UK", "Germany", "France"]),
            "year_founded": random.randint(1980, 2015),
            "status": "Operating",
            "management_quality_score": round(random.uniform(1, 5), 1),
            "loan_agreement_ids": []
        })
    return new_companies

def generate_new_loans(num_loans, max_existing_loan_id_num, all_company_ids):
    """Generates new loan data."""
    new_loans = []
    current_max_id_num = max_existing_loan_id_num

    loan_types = ["Term Loan", "Revolving Credit Facility", "Equipment Financing", "Real Estate Loan"]
    currencies = ["USD", "EUR", "GBP"]
    statuses = ["Active", "Performing"]
    seniority_levels = ["Senior Secured", "Senior Unsecured", "Subordinated"]
    economic_conditions = ["Stable", "Growth", "Recessionary"]

    for i in range(num_loans):
        current_max_id_num += 1
        loan_id = f"LOAN{current_max_id_num:04d}"
        company_id = random.choice(all_company_ids)
        principal_amount = random.randint(50000, 2000000)
        maturity_date = (datetime.now() + timedelta(days=random.randint(365, 365*7))).strftime('%Y-%m-%d')
        origination_date = (datetime.now() - timedelta(days=random.randint(0, 365*2))).strftime('%Y-%m-%d')

        new_loans.append({
            "loan_id": loan_id,
            "company_id": company_id,
            "loan_type": random.choice(loan_types),
            "principal_amount": principal_amount,
            "currency": random.choice(currencies),
            "interest_rate": round(random.uniform(0.02, 0.10), 4),
            "origination_date": origination_date,
            "maturity_date": maturity_date,
            "collateral_type": "Various Assets" if random.random() > 0.3 else "None",
            "collateral_value": int(principal_amount * random.uniform(0.8, 1.5)) if random.random() > 0.3 else 0,
            "covenants_compliance": "Compliant",
            "status": random.choice(statuses),
            "seniority_of_debt": random.choice(seniority_levels),
            "economic_condition_indicator": random.choice(economic_conditions)
        })
    return new_loans

def generate_financial_statements(max_existing_fs_id_num, company_ids, target_years):
    """Generates plausible financial statements for companies over specified years."""
    new_statements = []
    current_max_id_num = max_existing_fs_id_num

    for company_id in company_ids:
        base_revenue = random.randint(1000000, 50000000)
        base_assets = random.randint(5000000, 100000000)

        for year_idx, year in enumerate(target_years):
            current_max_id_num += 1
            statement_id = f"FS{current_max_id_num:03d}"

            growth_factor = random.uniform(0.85, 1.25)
            current_revenue = int(base_revenue * (growth_factor ** year_idx))
            cogs = int(current_revenue * random.uniform(0.4, 0.7))
            gross_profit = current_revenue - cogs
            operating_expenses = int(gross_profit * random.uniform(0.3, 0.6))
            net_income_before_interest_tax = gross_profit - operating_expenses # More precise name

            interest_expense = int(net_income_before_interest_tax * random.uniform(0.05, 0.2))
            # Ensure net_income calculation is based on income after interest for tax calc
            income_before_tax = net_income_before_interest_tax - interest_expense
            taxes = int(max(0, income_before_tax) * random.uniform(0.1, 0.3))
            net_income = income_before_tax - taxes

            current_assets = int(base_assets * (growth_factor ** year_idx))
            total_liabilities = int(current_assets * random.uniform(0.2, 0.7))
            total_equity = current_assets - total_liabilities

            new_statements.append({
                "statement_id": statement_id,
                "company_id": company_id,
                "statement_date": f"{year}-12-31",
                "statement_type": "Annual",
                "currency": "USD",
                "revenue": current_revenue,
                "cost_of_goods_sold": cogs,
                "gross_profit": gross_profit,
                "operating_expenses": operating_expenses,
                "interest_expense": interest_expense,
                "taxes": taxes,
                "net_income": net_income,
                "total_assets": current_assets,
                "total_liabilities": total_liabilities,
                "total_equity": total_equity,
                "cash_and_equivalents": int(current_assets * random.uniform(0.05, 0.2)),
                "accounts_receivable": int(current_assets * random.uniform(0.1, 0.3)),
                "inventory": int(current_assets * random.uniform(0.1, 0.3)),
                "property_plant_equipment": int(current_assets * random.uniform(0.2, 0.5)),
                "accounts_payable": int(total_liabilities * random.uniform(0.2, 0.5)),
                "short_term_debt": int(total_liabilities * random.uniform(0.1, 0.4)),
                "long_term_debt": int(total_liabilities * random.uniform(0.3, 0.7))
            })
    return new_statements

def generate_default_events(max_existing_de_id_num, loan_ids_for_default, all_loans_data):
    new_events = []
    current_max_id_num = max_existing_de_id_num

    loan_details = {l['loan_id']: l for l in all_loans_data}

    for loan_id in loan_ids_for_default:
        current_max_id_num += 1
        event_id = f"DE{current_max_id_num:03d}"

        loan_info = loan_details.get(loan_id)
        if not loan_info:
            print(f"Warning: Loan {loan_id} not found for default event generation. Skipping.")
            continue

        try:
            origination_date = datetime.strptime(loan_info["origination_date"], '%Y-%m-%d')
            maturity_date = datetime.strptime(loan_info["maturity_date"], '%Y-%m-%d')
            min_default_days = 30
            max_default_days = (maturity_date - origination_date).days + 90

            if min_default_days >= max_default_days:
                default_date_dt = maturity_date + timedelta(days=random.randint(1, 90))
            else:
                default_date_dt = origination_date + timedelta(days=random.randint(min_default_days, max_default_days))
            default_date = default_date_dt.strftime('%Y-%m-%d')
        except (ValueError, TypeError) as e: # Catch TypeError if dates are not strings
            print(f"Warning: Could not parse dates for loan {loan_id} ('{loan_info.get('origination_date')}', '{loan_info.get('maturity_date')}'). Error: {e}. Setting generic default date.")
            default_date = (datetime.now() + timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d')

        new_events.append({
            "event_id": event_id,
            "loan_id": loan_id,
            "company_id": loan_info["company_id"],
            "default_date": default_date,
            "reason_for_default": random.choice(["Missed Payment", "Covenant Breach", "Bankruptcy Filing"]),
            "amount_at_default": int(loan_info.get("principal_amount", loan_info.get("loan_amount", 0)) * random.uniform(0.8, 1.0)),
            "recovery_percentage": round(random.uniform(0.1, 0.7), 2)
        })
    return new_events

# --- File I/O Functions ---

def read_csv_file(filepath):
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return list(reader)
    except FileNotFoundError:
        print(f"File not found: {filepath}. Returning empty list for {filepath}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return []


def overwrite_csv_file(filepath, all_data, fieldnames):
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)


def read_json_file(filepath):
    try:
        with open(filepath, mode='r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {filepath}. Returning empty list for {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}. Returning empty list.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return []

def write_json_file(filepath, data):
    with open(filepath, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# --- Main Script Logic ---
def main():
    random.seed(42)

    companies_file = 'data/sample_companies.csv'
    loans_file = 'data/sample_loans.json'
    financial_statements_file = 'data/sample_financial_statements.json'
    default_events_file = 'data/sample_default_events.json'

    print("Reading existing data...")
    existing_companies = read_csv_file(companies_file)
    existing_loans = read_json_file(loans_file)
    existing_financial_statements = read_json_file(financial_statements_file)
    existing_default_events = read_json_file(default_events_file)

    print(f"Initial counts: Companies={len(existing_companies)}, Loans={len(existing_loans)}, FS={len(existing_financial_statements)}, Defaults={len(existing_default_events)}")

    max_comp_id_num = get_max_existing_id(existing_companies, 'company_id', 'COMP')
    max_loan_id_num = get_max_existing_id(existing_loans, 'loan_id', 'LOAN')
    max_fs_id_num = get_max_existing_id(existing_financial_statements, 'statement_id', 'FS')
    max_de_id_num = get_max_existing_id(existing_default_events, 'event_id', 'DE')

    print("\nGenerating new data...")
    new_companies_data = generate_new_companies(NUM_NEW_COMPANIES, max_comp_id_num)
    print(f"Generated {len(new_companies_data)} new companies.")

    all_company_ids_for_loans = [c['company_id'] for c in existing_companies] + [c['company_id'] for c in new_companies_data]
    if not all_company_ids_for_loans: # Safety check if all company loading failed
        print("Error: No company IDs available for loan generation. Exiting.")
        return

    new_loans_data = generate_new_loans(NUM_NEW_LOANS, max_loan_id_num, all_company_ids_for_loans)
    print(f"Generated {len(new_loans_data)} new loans.")

    temp_companies_map = {c['company_id']: c for c in existing_companies}
    for c in new_companies_data:
        temp_companies_map[c['company_id']] = c

    for loan in new_loans_data:
        company_id = loan['company_id']
        if company_id in temp_companies_map:
            if 'loan_agreement_ids' not in temp_companies_map[company_id] or not isinstance(temp_companies_map[company_id].get('loan_agreement_ids'), list):
                 # Handle cases where loan_agreement_ids might be a string from CSV read, or missing
                current_ids_str = temp_companies_map[company_id].get('loan_agreement_ids', '')
                if isinstance(current_ids_str, str) and current_ids_str:
                    # Attempt to parse if it looks like a list string, e.g. "['LOAN1', 'LOAN2']" or "LOAN1;LOAN2"
                    if current_ids_str.startswith('[') and current_ids_str.endswith(']'):
                        try:
                            # Try to make it valid JSON for parsing: replace single quotes if used
                            processed_ids_str = current_ids_str.replace("'", "\"")
                            parsed_ids = json.loads(processed_ids_str)
                            temp_companies_map[company_id]['loan_agreement_ids'] = parsed_ids
                        except json.JSONDecodeError:
                             # If parsing fails, treat as semi-colon separated or single ID (fallback)
                            temp_companies_map[company_id]['loan_agreement_ids'] = [s.strip() for s in current_ids_str.strip("[]'").split(';') if s.strip()]
                    elif ';' in current_ids_str: # Semicolon separated
                        temp_companies_map[company_id]['loan_agreement_ids'] = [s.strip() for s in current_ids_str.split(';') if s.strip()]
                    else: # Treat as a single ID if not empty and not list-like
                         temp_companies_map[company_id]['loan_agreement_ids'] = [current_ids_str.strip()]

                else: # Not a string or empty string, or already a list (though caught by outer if)
                    temp_companies_map[company_id]['loan_agreement_ids'] = []

            # Ensure it's a list before appending
            if not isinstance(temp_companies_map[company_id]['loan_agreement_ids'], list):
                 temp_companies_map[company_id]['loan_agreement_ids'] = [] # Should not happen if above logic is correct

            temp_companies_map[company_id]['loan_agreement_ids'].append(loan['loan_id'])
        else:
             print(f"Warning: Company ID {company_id} from new loan {loan['loan_id']} not found in company map.")

    updated_all_companies_data = list(temp_companies_map.values())

    new_company_ids_for_fs = [c['company_id'] for c in new_companies_data]
    new_fs_data = generate_financial_statements(max_fs_id_num, new_company_ids_for_fs, YEARS_FINANCIAL_STATEMENTS)
    print(f"Generated {len(new_fs_data)} new financial statements.")

    num_defaults_to_generate = max(1, int(len(new_loans_data) * random.uniform(0.2, 0.3)))
    loans_for_default_candidates = [l['loan_id'] for l in new_loans_data]

    if len(loans_for_default_candidates) < num_defaults_to_generate:
        selected_loan_ids_for_default = loans_for_default_candidates
    elif loans_for_default_candidates: # Ensure there are candidates
        selected_loan_ids_for_default = random.sample(loans_for_default_candidates, num_defaults_to_generate)
    else:
        selected_loan_ids_for_default = [] # No new loans to make default

    all_loans_for_default_gen = existing_loans + new_loans_data
    new_default_events_data = generate_default_events(max_de_id_num, selected_loan_ids_for_default, all_loans_for_default_gen)
    print(f"Generated {len(new_default_events_data)} new default events.")

    print("\nAppending new data...")
    updated_loans = existing_loans + new_loans_data
    updated_financial_statements = existing_financial_statements + new_fs_data
    updated_default_events = existing_default_events + new_default_events_data

    print("\nWriting updated data to files...")
    if updated_all_companies_data:
        all_keys = set()
        for c in updated_all_companies_data:
            all_keys.update(c.keys())

        # Define a preferred field order, making sure all keys are included
        preferred_order = [
            "company_id", "company_name", "industry", "country", "year_founded",
            "status", "management_quality_score", "loan_agreement_ids",
            # Add any fields that might have been in original CSV but not in new gen
            "industry_sector", "country_iso_code", "founded_date",
            "revenue_usd_millions", "subsidiaries", "suppliers", "customers",
            "financial_statement_ids"
        ]
        # Add any other keys not in preferred_order to the end
        company_fieldnames = preferred_order + sorted(list(all_keys - set(preferred_order)))


        csv_ready_companies = []
        for company_data in updated_all_companies_data:
            company_copy = company_data.copy()
            # Standardize loan_agreement_ids to semicolon-separated string for CSV
            if 'loan_agreement_ids' in company_copy and isinstance(company_copy['loan_agreement_ids'], list):
                company_copy['loan_agreement_ids'] = ";".join(sorted(list(set(company_copy['loan_agreement_ids'])))) # sort for consistency & unique

            for key in company_fieldnames:
                if key not in company_copy:
                    company_copy[key] = ''
            csv_ready_companies.append(company_copy)

        overwrite_csv_file(companies_file, csv_ready_companies, company_fieldnames)
        print(f"Overwritten {companies_file}")
    else:
        print(f"No company data to write to {companies_file}")

    write_json_file(loans_file, updated_loans)
    print(f"Overwritten {loans_file}")
    write_json_file(financial_statements_file, updated_financial_statements)
    print(f"Overwritten {financial_statements_file}")
    write_json_file(default_events_file, updated_default_events)
    print(f"Overwritten {default_events_file}")

    print("\nVerification: Record counts after updates...")
    final_companies = read_csv_file(companies_file)
    final_loans = read_json_file(loans_file)
    final_fs = read_json_file(financial_statements_file)
    final_defaults = read_json_file(default_events_file)

    print(f"Final counts: Companies={len(final_companies)}, Loans={len(final_loans)}, FS={len(final_fs)}, Defaults={len(final_defaults)}")

    if new_companies_data and final_companies:
        new_comp_id_check = new_companies_data[0]['company_id']
        if any(c.get('company_id') == new_comp_id_check for c in final_companies):
            print(f"Verified: New company ID {new_comp_id_check} found in {companies_file}")
        else:
            print(f"Verification FAILED: New company ID {new_comp_id_check} NOT found in {companies_file}")

if __name__ == "__main__":
    main()
