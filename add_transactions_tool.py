import json
import os

def add_transactions_to_json(transactions):
    """
    Appends a list of transaction dictionaries to 'transactions.json'.
    Each dict can have keys like:
        - "Amount"
        - "Currency"
        - "Category"
        - "Transaction Type"
        - "Balance after transaction"
        - "Merchant Name"
        - "Description"
    """
    file_path = "transactions.json"

    # If file doesn't exist, start with an empty list
    if not os.path.exists(file_path):
        existing_data = []
    else:
        # Read and parse existing JSON
        with open(file_path, "r") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []

    # Append the new transactions
    existing_data.extend(transactions)

    # Save back to the JSON file
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)