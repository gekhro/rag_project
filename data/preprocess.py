# write functions to preprocess the datasets in the datasets folder here
import json
import csv
import os
import re
import datasets
from pathlib import Path
import pandas as pd

def process_human_eval(input_file: str, output_file: str):
    """
    Process HumanEval dataset to match MBPP format.
    
    Args:
        input_file: Path to the HumanEval jsonl file
        output_file: Path to save the processed MBPP-format jsonl file
    """
    print(f"Processing HumanEval dataset from {input_file}")
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract the function name from entry_point
            function_name = data['entry_point']
            
            # Create problem statement from prompt
            text = data['prompt'].split('"""')[1].strip() if '"""' in data['prompt'] else data['prompt']
            
            # Combine prompt and solution for code
            full_code = data['prompt'].replace('"""' + text + '"""', '').strip() + '\n' + data['canonical_solution'].strip()
            full_code = full_code.strip()
            
            # Process test cases
            test_list = []
            if 'test_case_list' in data:
                test_list = data['test_case_list']
            
            # Create MBPP format entry
            mbpp_entry = {
                "text": text,
                "code": full_code,
                "task_id": f"HumanEval_{data['task_id'].replace('HumanEval/', '')}",
                "test_setup_code": "",
                "test_list": test_list,
                "challenge_test_list": []
            }
            
            processed_data.append(mbpp_entry)
    
    # Write processed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(processed_data)} HumanEval examples to {output_file}")
    return processed_data

def process_kaggle_csv(input_file: str, output_file: str):
    """
    Process Kaggle CSV dataset to match MBPP format.
    
    Args:
        input_file: Path to the CSV file
        output_file: Path to save the processed MBPP-format jsonl file
    """
    print(f"Processing Kaggle CSV dataset from {input_file}")
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            problem = row.get('Problem', '')
            code = row.get('Python Code', '')
            
            # Skip empty entries
            if not problem or not code:
                continue
            
            # Create test cases if possible by looking for expected output comments
            test_list = []
            code_lines = code.split('\n')
            for line in code_lines:
                if '# Expected output:' in line:
                    # Try to create a simple assertion test
                    output_text = line.split('# Expected output:')[1].strip()
                    if output_text and not output_text.startswith('[Error]'):
                        function_match = re.search(r'def\s+(\w+)\s*\(', code)
                        if function_match:
                            func_name = function_match.group(1)
                            # Simple assertion test - this is a best effort and might need manual review
                            test_list.append(f"assert {func_name}() == {output_text}")
            
            # Create MBPP format entry
            mbpp_entry = {
                "text": problem,
                "code": code,
                "task_id": f"Kaggle_{i + 1000}",  # Start IDs from 1000 with Kaggle prefix
                "test_setup_code": "",
                "test_list": test_list,
                "challenge_test_list": []
            }
            
            processed_data.append(mbpp_entry)
    
    # Write processed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(processed_data)} Kaggle examples to {output_file}")
    return processed_data

def process_mbpp(input_file: str, output_file: str):
    """
    Process original MBPP dataset.
    
    Args:
        input_file: Path to the MBPP jsonl file
        output_file: Path to save the processed MBPP-format jsonl file
    """
    print(f"Processing MBPP dataset from {input_file}")
    
    # Since this is already in the right format, we just need to read and write
    # But we'll validate each entry to ensure consistency
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Ensure all required fields are present
            mbpp_entry = {
                "text": data.get("text", ""),
                "code": data.get("code", ""),
                "task_id": data.get("task_id", ""),
                "test_setup_code": data.get("test_setup_code", ""),
                "test_list": data.get("test_list", []),
                "challenge_test_list": data.get("challenge_test_list", [])
            }
            
            # Only add if text and code are not empty
            if mbpp_entry["text"] and mbpp_entry["code"]:
                # Prefix the task_id if it's just a number to avoid conflicts
                if isinstance(mbpp_entry["task_id"], int):
                    mbpp_entry["task_id"] = f"MBPP_{mbpp_entry['task_id']}"
                processed_data.append(mbpp_entry)
    
    # Write processed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(processed_data)} MBPP examples to {output_file}")
    return processed_data

def process_extended_mbpp(input_file: str, output_file: str):
    """
    Process extended MBPP dataset.
    
    Args:
        input_file: Path to the extended MBPP jsonl file
        output_file: Path to save the processed MBPP-format jsonl file
    """
    print(f"Processing extended MBPP dataset from {input_file}")
    
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line_count += 1
            try:
                data = json.loads(line.strip())
                
                # Extract relevant fields and format according to MBPP
                # Adjust field names if they differ from standard MBPP
                
                # For extended MBPP, field names might be different
                # Common mappings that might exist in extended datasets
                text = data.get("text", data.get("problem", data.get("description", "")))
                code = data.get("code", data.get("solution", data.get("python_code", "")))
                task_id = data.get("task_id", data.get("id", f"MBPP_ET_{line_count}"))
                
                # Process test cases - might be in different formats
                test_list = data.get("test_list", data.get("tests", []))
                # Convert test_list to the right format if needed
                if isinstance(test_list, str):
                    test_list = [test_list]
                
                # Create MBPP format entry
                mbpp_entry = {
                    "text": text,
                    "code": code,
                    "task_id": f"MBPP_ET_{task_id}" if not str(task_id).startswith("MBPP_ET_") else task_id,
                    "test_setup_code": data.get("test_setup_code", ""),
                    "test_list": test_list,
                    "challenge_test_list": data.get("challenge_test_list", [])
                }
                
                # Only add if we have both text and code
                if mbpp_entry["text"] and mbpp_entry["code"]:
                    processed_data.append(mbpp_entry)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_count} in {input_file}")
                continue
    
    # Write processed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(processed_data)} extended MBPP examples to {output_file}")
    return processed_data

def process_datasets_vault(dataset_name: str, output_file: str):
    """
    Process datasets from Hugging Face's datasets library.
    
    Args:
        dataset_name: Name of the dataset in the Hugging Face library
        output_file: Path to save the processed MBPP-format jsonl file
    """
    print(f"Loading dataset {dataset_name} from Hugging Face datasets")
    
    try:
        # Load the dataset from Hugging Face
        dataset = datasets.load_dataset(dataset_name)
        
        # Determine which split to use (train, validation, test)
        split = list(dataset.keys())[0]  # Default to first available split
        
        print(f"Processing {dataset_name} (split: {split})")
        data = dataset[split]
        
        processed_data = []
        
        # Map dataset fields to MBPP format
        field_mappings = {
            'text': ['text', 'prompt', 'description', 'problem', 'task_description', 'question'],
            'code': ['code', 'solution', 'answer', 'python_code', 'completion'],
            'task_id': ['task_id', 'id', 'example_id'],
            'test_list': ['test_list', 'tests', 'test_cases', 'examples'],
        }
        
        # Process each example
        for i, example in enumerate(data):
            # Find text field
            text = ""
            for field in field_mappings['text']:
                if field in example:
                    text = example[field]
                    break
            
            # Find code field
            code = ""
            for field in field_mappings['code']:
                if field in example:
                    code = example[field]
                    break
            
            # Find task_id
            task_id = None
            for field in field_mappings['task_id']:
                if field in example:
                    task_id = example[field]
                    break
            
            if task_id is None:
                task_id = f"{dataset_name.replace('/', '_')}_{i}"
            
            # Find test cases
            test_list = []
            for field in field_mappings['test_list']:
                if field in example:
                    if isinstance(example[field], list):
                        test_list = example[field]
                    elif isinstance(example[field], str):
                        # Try to split string into list if it contains assertions
                        if "assert" in example[field]:
                            test_list = [line.strip() for line in example[field].split('\n') 
                                        if "assert" in line]
                        else:
                            test_list = [example[field]]
                    break
            
            # Create MBPP format entry
            mbpp_entry = {
                "text": text,
                "code": code,
                "task_id": f"HF_{task_id}",
                "test_setup_code": "",
                "test_list": test_list,
                "challenge_test_list": []
            }
            
            # Only add if we have both text and code
            if mbpp_entry["text"] and mbpp_entry["code"]:
                processed_data.append(mbpp_entry)
        
        # Write processed data to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in processed_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Processed {len(processed_data)} examples from {dataset_name} to {output_file}")
        return processed_data
    
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        return []

def process_devastator_kaggle(input_file: str, output_file: str):
    """
    Process Devastator Kaggle dataset to match MBPP format.
    In this dataset, the problem text is under 'question' column,
    solution code is under 'solutions' column, and expected inputs/outputs
    are in the 'input_output' column.
    
    Args:
        input_file: Path to the CSV file
        output_file: Path to save the processed MBPP-format jsonl file
    """
    print(f"Processing Devastator Kaggle dataset from {input_file}")
    processed_data = []
    
    try:
        # Try multiple approaches to read the CSV file
        try:
            # First approach: Use pandas with C engine and specify a large max_col_width
            # This may work if fields are large but not excessively so
            print("Attempting to read CSV with pandas C engine...")
            df = pd.read_csv(
                input_file, 
                encoding='utf-8',
                engine='c',
                on_bad_lines='warn'
            )
        except Exception as e1:
            print(f"First approach failed: {e1}")
            try:
                # Second approach: Try reading with Python engine and chunks
                print("Attempting to read CSV with Python engine and iterative processing...")
                # Increase the CSV field size limit first
                import sys
                from csv import field_size_limit
                # Increase the CSV field size limit to the maximum allowed value
                max_int = sys.maxsize
                while True:
                    try:
                        field_size_limit(max_int)
                        break
                    except OverflowError:
                        max_int = int(max_int/10)
                print(f"Set CSV field size limit to {max_int}")
                
                # Read in chunks and process each chunk
                chunk_size = 10  # Small chunks to handle large rows
                chunks = pd.read_csv(
                    input_file, 
                    encoding='utf-8',
                    engine='python',
                    chunksize=chunk_size,
                    error_bad_lines=False,
                    warn_bad_lines=True
                )
                
                # Combine chunks into a DataFrame
                df = pd.concat(chunks)
                
            except Exception as e2:
                print(f"Second approach failed: {e2}")
                # If both pandas approaches fail, try a more manual approach
                print("Attempting to read as text file and parse manually...")
                
                # Read lines and manually parse CSV
                from io import StringIO
                import csv
                
                # Read the whole file as text
                with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Split by newlines to get rows
                rows = content.split('\n')
                if len(rows) < 2:
                    raise ValueError("File has too few rows or invalid format")
                
                # Get header row
                header = next(csv.reader(StringIO(rows[0])))
                
                # Create empty DataFrame with the headers
                df = pd.DataFrame(columns=header)
                
                # Find column indices for the fields we need
                try:
                    question_idx = header.index('question')
                    solutions_idx = header.index('solutions')
                    input_output_idx = header.index('input_output') if 'input_output' in header else -1
                except ValueError as e:
                    print(f"Could not find required columns: {e}")
                    raise
                
                # Process each row manually
                for i, row_text in enumerate(rows[1:], 1):
                    if not row_text.strip():
                        continue  # Skip empty rows
                    
                    try:
                        # Parse as CSV row, but handle quotes properly
                        row_data = next(csv.reader(StringIO(row_text)))
                        
                        # Skip if row doesn't have enough columns
                        if len(row_data) <= max(question_idx, solutions_idx):
                            print(f"Skipping row {i}: not enough columns")
                            continue
                        
                        # Extract data
                        row_dict = {
                            'question': row_data[question_idx],
                            'solutions': row_data[solutions_idx],
                        }
                        
                        if input_output_idx >= 0 and len(row_data) > input_output_idx:
                            row_dict['input_output'] = row_data[input_output_idx]
                        
                        # Add to dataframe
                        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
                    except Exception as row_error:
                        print(f"Error processing row {i}: {row_error}")
        
        print(f"Successfully loaded CSV with {len(df)} rows")
        
        # Process each row in the DataFrame
        for i, row in df.iterrows():
            # Extract data from row
            try:
                question = str(row.get('question', '')) if not pd.isna(row.get('question', '')) else ''
                solution = str(row.get('solutions', '')) if not pd.isna(row.get('solutions', '')) else ''
                input_output = str(row.get('input_output', '')) if not pd.isna(row.get('input_output', '')) else ''
                
                # Skip empty entries
                if not question or not solution:
                    continue
                
                # Create test cases if possible by extracting them from the solution or question
                test_list = []
                
                # First try to extract test cases from input_output column
                if input_output:
                    # Try to parse the input_output data which might contain test cases
                    function_match = re.search(r'def\s+(\w+)\s*\(', solution)
                    if function_match:
                        func_name = function_match.group(1)
                        
                        # Look for patterns like "Input: X, Output: Y" or similar in input_output
                        input_output_pairs = re.findall(r'(?:Input|input|In|in)(?:\s*:|\s*=|\s*>|\s*)\s*(.*?)(?:,|;|\n)?\s*(?:Output|output|Out|out)(?:\s*:|\s*=|\s*>|\s*)\s*(.*?)(?:$|\n|;|,)', input_output, re.DOTALL)
                        
                        if input_output_pairs:
                            for input_val, output_val in input_output_pairs:
                                input_val = input_val.strip()
                                output_val = output_val.strip()
                                
                                # Try to create a test case based on the input-output pair
                                if input_val and output_val:
                                    # For simple cases, create direct assertion
                                    test_list.append(f"assert {func_name}({input_val}) == {output_val}")
                        else:
                            # If no structured pairs found, look for test examples
                            examples = re.findall(r'(?:Example|example|Test|test)(?:\s*:|\s*\d+:|\s*\d+\s*:|:)\s*(.*?)(?:$|\n|(?:Example|example|Test|test))', input_output + "\nExample", re.DOTALL)
                            
                            for example in examples:
                                # Try to extract input and output from example text
                                io_match = re.search(r'(?:Input|input|In|in)(?:\s*:|\s*=|\s*>|\s*)\s*(.*?)(?:,|;|\n)?\s*(?:Output|output|Out|out)(?:\s*:|\s*=|\s*>|\s*)\s*(.*?)(?:$|\n|;|,)', example, re.DOTALL)
                                if io_match:
                                    input_val = io_match.group(1).strip()
                                    output_val = io_match.group(2).strip()
                                    
                                    if input_val and output_val:
                                        test_list.append(f"assert {func_name}({input_val}) == {output_val}")
                
                # Also check for example outputs or test comments in the solution if we don't have tests yet
                if not test_list:
                    code_lines = solution.split('\n')
                    for line in code_lines:
                        if any(test_indicator in line.lower() for test_indicator in 
                            ['# test', '# expected', '# output', '# example']):
                            # Try to extract test information
                            function_match = re.search(r'def\s+(\w+)\s*\(', solution)
                            if function_match:
                                func_name = function_match.group(1)
                                # If there's an expected output value in the comment
                                output_match = re.search(r':\s*(.*?)$', line)
                                if output_match:
                                    expected_output = output_match.group(1).strip()
                                    if expected_output:
                                        test_list.append(f"assert {func_name}() == {expected_output}")
                
                # Create MBPP format entry
                mbpp_entry = {
                    "text": question,
                    "code": solution,
                    "task_id": f"Devastator_{i + 2000}",  # Start IDs from 2000 with Devastator prefix
                    "test_setup_code": "",
                    "test_list": test_list,
                    "challenge_test_list": []
                }
                
                processed_data.append(mbpp_entry)
                
                # Progress reporting
                if (i+1) % 100 == 0:
                    print(f"Processed {i+1} rows so far")
                    
            except Exception as row_error:
                print(f"Error processing row {i}: {row_error}")
                continue
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
    
    # Write processed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(processed_data)} Devastator Kaggle examples to {output_file}")
    return processed_data

def main():
    """
    Process all datasets and save them to the Processed_Datasets folder.
    """
    # Create output directory if it doesn't exist
    output_dir = "Processed_Datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # # Process original MBPP dataset
    # process_mbpp(
    #     input_file="Datasets/mbpp.jsonl",
    #     output_file=f"{output_dir}/processed_mbpp.jsonl"
    # )
    
    # # Process extended MBPP dataset
    # process_extended_mbpp(
    #     input_file="Datasets/MBPP_ET.jsonl",
    #     output_file=f"{output_dir}/processed_mbpp_extended.jsonl"
    # )
    
    # # Process HumanEval dataset
    # process_human_eval(
    #     input_file="Datasets/HumanEval_ET.jsonl",
    #     output_file=f"{output_dir}/processed_human_eval.jsonl"
    # )
    
    # # Process Kaggle CSV dataset
    # process_kaggle_csv(
    #     input_file="Datasets/ProblemSolutionPythonV3.csv",
    #     output_file=f"{output_dir}/processed_kaggle.jsonl"
    # )
    
    # Process Devastator Kaggle dataset
    process_devastator_kaggle(
        input_file="Datasets/kaggle_devastator_train.csv",
        output_file=f"{output_dir}/processed_devastator_kaggle.jsonl"
    )
    
    # # Process datasets from HuggingFace
    # # Examples of coding datasets to process from HuggingFace
    # hf_datasets = [
    #     "codeparrot/apps",  # APPS dataset
    #     "nuprl/mbpp-sanitized",  # Sanitized MBPP
    # ]
    
    # for dataset_name in hf_datasets:
    #     safe_name = dataset_name.replace("/", "_")
    #     process_datasets_vault(
    #         dataset_name=dataset_name,
    #         output_file=f"{output_dir}/processed_{safe_name}.jsonl"
    #     )
    
    print(f"All datasets have been processed and saved to {output_dir}/")

if __name__ == "__main__":
    main()


