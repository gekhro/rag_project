import json

input_file = "./data/augmented_data.jsonl"
output_file = "./data/cleaned_augmented_final_data.jsonl"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        # Create new dictionary with only transformed fields
        new_data = {
            "text": data["transformed_text"],
            "code": data["transformed_code"],
        }
        # Write the new data to output file
        outfile.write(json.dumps(new_data) + "\n")
