import os
import math
import zstandard as zstd

def split_and_compress_csv(input_file, output_dir, compression_level=22):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine the total number of lines in the input file
    with open(input_file, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for line in infile)

    # Calculate the approximate number of lines per file
    lines_per_file = math.ceil(total_lines / 3)

    file_number = 0
    line_count = 0
    output_file = None
    output_compressed_file = None
    compressor = zstd.ZstdCompressor(level=compression_level)

    with open(input_file, 'r', encoding='utf-8') as infile:
        header = infile.readline()
        for line in infile:
            if line_count % lines_per_file == 0:
                if output_file:
                    output_file.close()
                    output_compressed_file.close()
                file_number += 1
                output_filename = os.path.join(output_dir, f'split_{file_number}.csv')
                compressed_filename = os.path.join(output_dir, f'split_{file_number}.csv.zst')

                output_file = open(output_filename, 'w', encoding='utf-8')
                output_compressed_file = open(compressed_filename, 'wb')

                output_file.write(header)
                output_compressed_file.write(compressor.compress(header.encode('utf-8')))

            output_file.write(line)
            output_compressed_file.write(compressor.compress(line.encode('utf-8')))
            line_count += 1

        if output_file:
            output_file.close()
            output_compressed_file.close()

    print(f"File split into {file_number} smaller files and compressed in '{output_dir}'.")
    return [os.path.join(output_dir, f'split_{i}.csv.zst') for i in range(1, file_number + 1)]

def validate_splits(original_file, split_files):
    total_lines_original = sum(1 for _ in open(original_file, 'r', encoding='utf-8'))
    total_lines_split = 0

    decompressor = zstd.ZstdDecompressor()
    for split_file in split_files:
        with open(split_file, 'rb') as fin:
            decompressed = decompressor.decompress(fin.read())
            total_lines_split += len(decompressed.decode('utf-8').splitlines()) - 1  # Subtracting 1 for the header

    if total_lines_original == total_lines_split:
        print("Validation successful: The sum of all split files equals the original file.")
    else:
        print(f"Validation failed: Original file has {total_lines_original} lines, but split files sum to {total_lines_split} lines.")

def decompress_and_merge(split_files, output_file):
    decompressor = zstd.ZstdDecompressor()
    with open(output_file, 'w', encoding='utf-8') as fout:
        header_written = False
        for split_file in split_files:
            with open(split_file, 'rb') as fin:
                decompressed = decompressor.decompress(fin.read()).decode('utf-8')
                lines = decompressed.splitlines()
                if not header_written:
                    fout.write(lines[0] + '\n')  # Write header from the first file
                    header_written = True
                fout.write('\n'.join(lines[1:]) + '\n')  # Write the rest of the lines, skipping the header

    print(f"Files decompressed and merged into '{output_file}'.")

# Example usage
# Directory path
directory = 'BillboardExplain/data/group_0'

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
input_file = os.path.join(directory, 'group_0_results.csv')

# output_dir = 'output_chunks'
# Split and compress the file
split_files = split_and_compress_csv(input_file, directory)

# Validate the split files
validate_splits(input_file, split_files)

# Reverse operation: Decompress and merge the split files back into a large file
# output_merged_file = 'merged_large_file.csv'
# decompress_and_merge(split_files, output_merged_file)
