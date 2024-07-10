input_files = []
for i in range(100):
    input_files.append(f'datagen/data_part_{i}.txt')

output_file = 'merged_file.txt'

with open(output_file, 'w') as outfile:
    for file_name in input_files:
        with open(file_name, 'r') as infile:
            outfile.write(infile.read())
