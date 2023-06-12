import os
from pathlib import Path

git_dir = Path(".")  # set your git directory path here
output_file = "output.txt"

with open(output_file, "w") as f:
    for file in git_dir.glob("**/*.py"):
        f.write(f'<file path={str(file)}>\n')
        with open(file, "r") as py_file:
            f.write(py_file.read())
        f.write("\n</file>\n")