import json
import os
import argparse

def combine_jsons(input_folder, output_file):
    final_dic = {}
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        
        if file.endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                key = os.path.splitext(file)[0]
                final_dic[key] = data.get(key, {})
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    with open(output_file, 'w') as out_f:
        json.dump(final_dic, out_f, indent=0)

    print(f"Combined JSON saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine JSON files into one.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing JSON files to combine.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the combined JSON file.")
    args = parser.parse_args()

    combine_jsons(args.input_folder, args.output_file)
