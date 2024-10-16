#!/bin/bash

DATA_DIR="/mnt/c/Users/Imesh/Desktop/data"
INPUT_FILE="input.txt"  
REG_FILE="flux.reg"
BKG_FILE="bkg.reg"
PYTHON_SCRIPT="measure_flux.py"  

OUTPUT_FILE="spectral_indices.txt"

echo "Source,Alpha,Alpha_Error" > $OUTPUT_FILE

for SOURCE_DIR in "$DATA_DIR"/*; do
    # Check if it is a directory
    if [ -d "$SOURCE_DIR" ]; then
        # Check if the directory contains the required FITS and region files
        FITS_FILES=($(ls "$SOURCE_DIR"/sub_band_C_block*.fits 2> /dev/null))
        REG_FILE_PATH="$SOURCE_DIR/$REG_FILE"
        BKG_FILE_PATH="$SOURCE_DIR/$BKG_FILE"
        
        if [ -f "$REG_FILE_PATH" ] && [ -f "$BKG_FILE_PATH" ] && [ ${#FITS_FILES[@]} -gt 0 ]; then
            echo "Processing source: $(basename "$SOURCE_DIR")"

            echo "${FITS_FILES[@]}" > "$SOURCE_DIR/$INPUT_FILE"
            
            python3 $PYTHON_SCRIPT --input_file "$SOURCE_DIR/$INPUT_FILE" --output_file "$OUTPUT_FILE" --reg_file "$REG_FILE_PATH" --bkg_file "$BKG_FILE_PATH"
        else
            echo "Skipping directory: $(basename "$SOURCE_DIR") - missing required files"
        fi
    fi
done

echo "Processing complete. Spectral indices saved to $OUTPUT_FILE"

