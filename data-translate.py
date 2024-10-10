import json

# Function to read and process data from the file
def process_data_file(file_path):
    output = []
    
    # Open the file and read line by line
    with open(file_path, "r") as file:
        for line in file:
            # Ignore comment lines or lines starting with '%'
            if line.startswith('%') or not line.strip():
                continue
            
            # Split the line by spaces to extract the fields
            fields = line.split()
            
            # Ensure the line has enough fields to process (assuming 7 or more fields)
            if len(fields) >= 6:
                t = float(fields[0])
                x = float(fields[1])
                y = float(fields[2])
                pres = float(fields[3])
                u = float(fields[4])
                v = float(fields[5])

                # Append the data row to the output
                output.append([t, x, y, pres, u, v])
    
    return output

# File path to your dataset
file_path = "your.file.path/name.if" # If it's at the same spot as this file.

# Process the data from the file
output_data = process_data_file(file_path)

# Convert the output to JSON format without a header
json_output = json.dumps(output_data, indent=4)

# Write the JSON output to a file
with open("output_data.json", "w") as json_file: ## You can change the name to data.py directly from here, or remember to do it when you're importing it to the /data/ folder.
    json_file.write(json_output)

# Print a message to indicate success
print("Data successfully converted to JSON and saved as 'output_data.json'")
