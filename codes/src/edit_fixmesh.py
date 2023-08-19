import re

# Define the code block you want to search for
code_block_to_search = r'''
if __package__ or "." in __name__:
    from . import _fixmesh
else:
    import _fixmesh
'''

# Define the replacement code
replacement_code = '\nimport _fixmesh\n'

# Specify the input Python file
input_file = 'fixmesh.py'

# Read the input file
with open(input_file, 'r') as file:
    file_contents = file.read()

# Use regular expressions to find and replace the code block
new_contents = re.sub(re.escape(code_block_to_search), replacement_code, file_contents)

# Write the modified contents back to the file
with open(input_file, 'w') as file:
    file.write(new_contents)

print("Code block replaced successfully.")
