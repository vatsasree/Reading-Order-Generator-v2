#!/bin/bash

# Define input folder, output folder, and final output JSON file
# input_folder_1="/ssd_scratch/sreevatsa/Reading-Order-Dataset/extended_dataset"
# input_folder_1="/ssd_scratch/sreevatsa/Reading-Order-Dataset/extended_dataset_br"
# input_folder_1="/ssd_scratch/sreevatsa/Reading-Order-Dataset/yojana_telugu_mal"
# input_folder_1="/ssd_scratch/sreevatsa/Reading-Order-Dataset/challenging"
# input_folder_1="/ssd_scratch/sreevatsa/Reading-Order-Dataset/textbooks"
# input_folder_1="/ssd_scratch/sreevatsa/DATASETS/RVL_CDIP/sampled_RO"
# input_folder_1="/ssd_scratch/sreevatsa/Reading-Order-Dataset/subset"
input_folder_1="/ssd_scratch/sreevatsa/SURYA_Sreevatsa/annotation_pipeline_test_images_GRs/images"

input_folder_2="/ssd_scratch/sreevatsa/Reading-Order-Dataset/final_images_combined_extended"

# output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/jsons_1191"
# output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/results_1191"
# output_file="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/reading_order_results_1191.json"

# output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/jsons_yoj_te_mal"
# output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/results_yoj_te_mal"
# output_file="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/reading_order_results_yoj_tel_mal.json"

# output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/jsons_prima_challenging"
# output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/results_prima_challenging"
# output_file="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/reading_order_results_challenging_prima.json"

# output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/tests"
# output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/tests"
# output_file="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/reading_order_results_rvlcdip.json"

# output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/jsons_extended_brs_v2_cpfix"
# output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/results_extended_brs_v2_cpfix"
# output_file="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/reading_order_results_ext_br_v2_cpfix.json"

# output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/jsons_prima_challenging_v2_tests"
# output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/results_prima_challenging_v2_tests"
# output_file="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/reading_order_results_challenging_prima_v2.json"

# output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/jsons_sampled_rvlcdip_v2"
# output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/results_sampled_rvlcdip_v2"
# output_file="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/reading_order_results_sampled_rvlcdip_v2.json"

# output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/jsons_textbooks_v2_cpfix"
# output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/results_textbooks_v2_cpfix"
# output_file="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/reading_order_results_textbooks_v2_cpfix.json"

output_folder="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/GRs_TD_cpfix_jsons"
output_folder_images="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/GRs_TD_cpfix_images"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Clear the existing combined dictionary file if it exists
> "$output_file"

process_images() {
    input_folder=$1

    # Process each image in the input folder
    for img_path in "$input_folder"/*; do
        # Extract the base name of the image (without extension)
        img_name=$(basename "$img_path")
        img_name_no_ext="${img_name%.*}"

        # Check if a JSON file with the same base name exists in the output folder
        json_path="$output_folder/$img_name_no_ext.json"
        if [ ! -f "$json_path" ]; then
            echo "Processing $img_path"
            python3 reading_order.py --input "$img_path" --output_folder "$output_folder" --output_folder_images "$output_folder_images"
        else
            echo "Skipping $img_path; JSON already exists."
        fi
    done
}

# process_images() {
#     input_folder=$1

#     # Process each image in the input folder
#     for img_path in "$input_folder"/*; do
#         # Extract the base name of the image (without extension)
#         img_name=$(basename "$img_path")
#         img_name_no_ext="${img_name%.*}"

#         # Process the image only if the filename contains any of the specified patterns
#         if [[ "$img_name" =~ (128|130|213|400|421|636|644|717|720|728|730|737) ]]; then
#             # Check if a JSON file with the same base name exists in the output folder
#             json_path="$output_folder/$img_name_no_ext.json"
#             if [ ! -f "$json_path" ]; then
#                 echo "Processing $img_path"
#                 python3 reading_order.py --input "$img_path" --output_folder "$output_folder" --output_folder_images "$output_folder_images"
#             else
#                 echo "Skipping $img_path; JSON already exists."
#             fi
#         else
#             echo "Skipping $img_path; filename does not contain required pattern."
#         fi
#     done
# }

# Process images from both folders
process_images "$input_folder_1"
# process_images "$input_folder_2"

# # Combine the generated JSON files into a single JSON file
# python3 combine_json.py --input_folder "$output_folder" --output_file "$output_file"

echo "Results saved to $output_file"
