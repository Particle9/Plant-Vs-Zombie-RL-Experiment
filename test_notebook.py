import nbformat
with open("comparison.ipynb", "r") as f:
    nb = nbformat.read(f, as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        print(f"--- Cell {i} ---")
        print(cell.source[:200])
