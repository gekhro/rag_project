import json
import os
import random
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "gpt-4o-mini"


def count_tokens(text):
    """
    Count tokens using tiktoken library.

    Args:
        text: The text to count tokens for
        model: Model name to use appropriate encoding (if None, uses cl100k_base)

    Returns:
        int: Number of tokens
    """
    encoding_name = "cl100k_base"

    encoding = tiktoken.get_encoding(encoding_name)

    return len(encoding.encode(text))


def transform_entry_with_retry(client, model, config, entry, max_retries=3):
    while True:
        retry_count = 0
        while retry_count < max_retries:
            try:
                result = transform_single_entry(client, model, config, entry)
                if result:
                    return result

                # If transform_entry returns None, wait and retry
                retry_count += 1
                if retry_count < max_retries:
                    # Exponential backoff with jitter
                    wait_time = (2**retry_count) + random.uniform(0, 1)
                    print(
                        f"Retrying in {wait_time:.1f} seconds (attempt {retry_count + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)

            except Exception as e:
                if "rate_limit" in str(e).lower():
                    retry_count += 1
                    if retry_count < max_retries:
                        # Longer wait for rate limits
                        wait_time = (2**retry_count) * 2 + random.uniform(0, 1)
                        print(
                            f"Rate limit hit. Retrying in {wait_time:.1f} seconds (attempt {retry_count + 1}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(
                            "Max retries reached for rate limit, waiting 30 seconds before starting fresh..."
                        )
                        time.sleep(30)  # Long wait before resetting retry count
                        retry_count = 0  # Reset retry count to try again
                else:
                    print(f"Unhandled error: {str(e)}")
                    raise


def transform_single_entry(client, model, config, entry):
    # Check if code is over 3000 tokens
    code_tokens = count_tokens(entry["code"])
    is_large_code = code_tokens > 3000

    if is_large_code:
        print(f"Code is large ({code_tokens} tokens), only transforming text")
        system_prompt = "You are a code transformation assistant. Your task is to transform text into a simplified format."

        # Format the single entry but only include the text portion
        entry_text = f"Entry text:\n{entry['text']}"

        user_prompt = (
            "Transform the following text entry into a simplified format.\n\n"
            f"{entry_text}\n\n"
            "Rules for text transformation:\n"
            "- Keep it VERY short and simple (max 20 words)\n"
            "- Use natural, command-like statements\n"
            "- Start with action verbs (sort, find, get, make, etc.)\n"
            "- Focus only on the main action\n"
            "- Avoid technical jargon\n"
            "- No question marks\n\n"
            "Output your result in the following JSON format exactly:\n"
            '{\n  "text": "transformed text here"\n}\n'
        )
    else:
        system_prompt = (
            "You are a code transformation assistant. Your task is to transform code and text into a simplified format, "
            "following specific rules and outputting the results in a JSON array format."
        )

        # Format the single entry
        entry_text = f"Entry:\n{json.dumps(entry, indent=2)}"

        user_prompt = (
            "Transform the following code and text entry into a simplified format.\n\n"
            f"{entry_text}\n\n"
            "Rules for code transformation:\n"
            "- Remove all comments and links\n"
            "- Convert 'def' functions to direct lambda or function expressions\n"
            "- Convert 'class' definitions to direct object creation\n"
            "- Keep only pure Python code, no docstrings or decorators\n"
            "- Make code as concise as possible\n\n"
            "Rules for text transformation:\n"
            "- Keep it VERY short and simple (max 20 words)\n"
            "- Use natural, command-like statements\n"
            "- Start with action verbs (sort, find, get, make, etc.)\n"
            "- Focus only on the main action\n"
            "- Avoid technical jargon\n"
            "- No question marks\n\n"
            "Output your result in the following JSON format exactly:\n"
            '{\n  "text": "transformed text here",\n  "code": "transformed code here"\n}\n'
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content
        print("\nAPI Response:")
        print("=" * 80)
        print(response_text)
        print("=" * 80)

        try:
            transformed_entry = json.loads(response_text)
            print("\nParsed JSON structure:")
            print(json.dumps(transformed_entry, indent=2))
            print(f"Type: {type(transformed_entry)}")
            if isinstance(transformed_entry, dict) and "results" in transformed_entry:
                transformed_entry = transformed_entry["results"]
                print("\nFound results object inside response")
        except json.JSONDecodeError as e:
            print(f"\nJSON parsing error: {str(e)}")
            return None

        if not isinstance(transformed_entry, dict):
            print(f"\nError: Expected dict but got {type(transformed_entry)}")
            return None

        print(f"\nValidating entry:")
        print(f"Transformed data: {json.dumps(transformed_entry, indent=2)}")

        if is_large_code:
            # For large code, we only expect text transformation
            if "text" not in transformed_entry:
                print(f"Error: Entry missing required 'text' field")
                return None

            result = {
                "original_text": entry["text"],
                "original_code": entry["code"],
                "transformed_text": transformed_entry["text"],
                "transformed_code": entry[
                    "code"
                ],  # Keep original code for large entries
            }
        else:
            # For normal entries, verify both text and code
            if "text" not in transformed_entry or "code" not in transformed_entry:
                print(f"Error: Entry missing required fields")
                return None

            result = {
                "original_text": entry["text"],
                "original_code": entry["code"],
                "transformed_text": transformed_entry["text"],
                "transformed_code": transformed_entry["code"],
            }

        # Validate result
        if all(result.values()):
            return result
        return None

    except json.JSONDecodeError as e:
        print(f"\nError parsing response: {response_text}")
        print(f"JSON error: {str(e)}")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return None


def augment_data(
    input_file="combined_data.jsonl",
    output_file="augmented_data.jsonl",
    start_idx=0,
    end_idx=None,
    checkpoint_file="checkpoint.txt",
):
    """
    Augment data by transforming entries in the input JSONL file.

    Args:
        input_file: Input JSONL file path (relative to data directory)
        output_file: Output JSONL file path (relative to data directory)
        start_idx: Starting index for processing (0-based)
        end_idx: Ending index for processing (exclusive), None for all entries
        checkpoint_file: File to save progress (relative to data directory)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    model = MODEL_NAME
    config = None

    # Ensure paths are relative to data directory
    data_dir = Path(__file__).parent
    input_path = data_dir / input_file
    output_path = data_dir / output_file
    checkpoint_path = data_dir / checkpoint_file

    # Load checkpoint if exists
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            start_idx = int(f.read().strip())
            print(f"Resuming from index {start_idx}")

    processed_count = 0
    total_entries = sum(1 for _ in open(input_path))

    if end_idx is None:
        end_idx = total_entries

    print(f"Processing entries {start_idx} to {end_idx} from {input_file}")
    print(f"Total entries in file: {total_entries}")

    # Process entries one by one and save immediately
    try:
        with open(input_path, "r") as infile, open(
            output_path, "a" if start_idx > 0 else "w"
        ) as outfile:
            # Skip to start_idx
            for _ in range(start_idx):
                next(infile)

            for i, line in enumerate(infile, start=start_idx + 1):
                if i > end_idx:
                    break

                try:
                    entry = json.loads(line)
                    print(f"\nProcessing entry {i}/{end_idx}...")

                    while True:  # Keep trying until we get a successful result
                        try:
                            result = transform_entry_with_retry(
                                client, model, config, entry
                            )
                            if result:
                                # Save the result immediately
                                json.dump(result, outfile)
                                outfile.write("\n")
                                outfile.flush()

                                processed_count += 1
                                print(f"Transformed and saved entry {i}")

                                # Update checkpoint after successful save
                                with open(checkpoint_path, "w") as f:
                                    f.write(str(i))
                                break
                            else:
                                print(
                                    "Failed to transform entry, retrying after 10 seconds..."
                                )
                                time.sleep(10)
                        except KeyboardInterrupt:
                            print("\nUser interrupted. Progress saved in checkpoint.")
                            return
                        except Exception as e:
                            print(f"Error processing entry: {str(e)}")
                            print("Retrying after 10 seconds...")
                            time.sleep(10)

                except json.JSONDecodeError as e:
                    print(f"Error parsing input line {i}: {str(e)}")
                    continue

        if checkpoint_path.exists():
            checkpoint_path.unlink()

    except KeyboardInterrupt:
        print("\nUser interrupted. Progress saved in checkpoint.")
        return
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print(f"Progress saved in checkpoint: {checkpoint_path}")
        raise

    print(f"\nProcessed {processed_count} entries. Results saved to {output_file}")
    print(f"Progress: {processed_count}/{end_idx - start_idx} entries processed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment data using DeepSeek API")
    parser.add_argument(
        "--input", default="combined_data.jsonl", help="Input JSONL file"
    )
    parser.add_argument(
        "--output", default="augmented_data.jsonl", help="Output JSONL file"
    )
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument(
        "--end", type=int, default=None, help="Ending index (exclusive)"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoint.txt", help="Checkpoint file"
    )

    args = parser.parse_args()

    augment_data(
        input_file=args.input,
        output_file=args.output,
        start_idx=args.start,
        end_idx=args.end,
        checkpoint_file=args.checkpoint,
    )
