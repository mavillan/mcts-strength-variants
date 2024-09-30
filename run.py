import papermill as pm
from pathlib import Path
import subprocess
import typer

def remove_output_cells(notebook_path:str) -> None:
    command = f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_path}"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, _ = process.communicate()
    print(output)
    
def run_notebook(notebook_path:str, overwrite:bool = True) -> None:
    notebook_path = Path(notebook_path)
    workdir = notebook_path.parent
    if not overwrite:
        output_path = notebook_path.parent / "outputs"
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / notebook_path.name
    else:
        output_path = notebook_path

    remove_output_cells(str(notebook_path))
    pm.execute_notebook(
        input_path=str(notebook_path), 
        output_path=str(output_path),
        cwd=str(workdir),
    )

if __name__ == "__main__":
    typer.run(run_notebook)