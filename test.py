import pandas as pd
import shelve

# Read the Excel file into a pandas DataFrame
dframe = pd.read_excel('data1.xlsx')

# Select the relevant columns
data = dframe[["Canonical Link", "Title", "About This Item"]]

# Convert the DataFrame to a list of dictionaries
data_dict_list = data.to_dict(orient='records')

# Open a shelve database
with shelve.open('data1') as excel_database_shelve:
    # Store each entry in the shelve database with a unique key
    for idx, row in enumerate(data_dict_list):
        key = f"row_{idx}"
        excel_database_shelve[key] = row

    # Function to search for product information by partial title
    def search_by_partial_title(partial_title):
        results = []
        for key in excel_database_shelve:
            product = excel_database_shelve[key]
            if partial_title.lower() in product["Title"].lower():
                results.append(product["Canonical Link"])
        return results

    # Example usage
    partial_title_to_search = 'NYX PROFESSIONAL MAKEUP Epic Ink Liner, Waterproof Liquid Eyeliner - Black, Vegan Formula'  # Replace with the partial title you are searching for
    matching_products = search_by_partial_title(partial_title_to_search)

    if matching_products:
        for product in matching_products:
            print(f"Product Info: {product}")
    else:
        print(f"No products found with title containing '{partial_title_to_search}'.")
