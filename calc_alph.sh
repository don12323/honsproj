#!/bin/bash

# Set the path to your data directory and region files
DATA_DIR="/mnt/c/Users/Imesh/Desktop/data"
INPUT_FILE="input.txt"  # This file will be generated for each source
REG_FILE="flux.reg"
BKG_FILE="bkg.reg"
PYTHON_SCRIPT="measure_flux.py"  # Your modified Python script from before

# Output file for the spectral indices
OUTPUT_FILE="spectral_indices.txt"

# Create or clear the output file
echo "Source,Alpha,Alpha_Error" > $OUTPUT_FILE

# Iterate over each directory in the data directory
for SOURCE_DIR in "$DATA_DIR"/*; do
    # Check if it is a directory
    if [ -d "$SOURCE_DIR" ]; then
        # Check if the directory contains the required FITS and region files
        FITS_FILES=($(ls "$SOURCE_DIR"/sub_band_C_block*.fits 2> /dev/null))
        REG_FILE_PATH="$SOURCE_DIR/$REG_FILE"
        BKG_FILE_PATH="$SOURCE_DIR/$BKG_FILE"
        
        # Proceed only if FITS files and both region files exist
        if [ -f "$REG_FILE_PATH" ] && [ -f "$BKG_FILE_PATH" ] && [ ${#FITS_FILES[@]} -gt 0 ]; then
            echo "Processing source: $(basename "$SOURCE_DIR")"

            # Create the input.txt file containing the list of FITS files for this source
            echo "${FITS_FILES[@]}" > "$SOURCE_DIR/$INPUT_FILE"
            
            # Call the Python script to process this source
            python3 $PYTHON_SCRIPT --input_file "$SOURCE_DIR/$INPUT_FILE" --output_file "$OUTPUT_FILE" --reg_file "$REG_FILE_PATH" --bkg_file "$BKG_FILE_PATH"
        else
            echo "Skipping directory: $(basename "$SOURCE_DIR") - missing required files"
        fi
    fi
done

echo "Processing complete. Spectral indices saved to $OUTPUT_FILE"

