import subprocess
import sys
import os

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running '{command}': {e}")
        sys.exit(1)

def main():
    # Create virtual environment
    run_command("python -m venv shallow_sa_env")

    # Determine the activation script based on the OS
    if sys.platform == "win32":
        activate_script = ".venv\\Scripts\\activate"
    else:
        activate_script = "source .venv/bin/activate"

    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("""
numpy
scipy
matplotlib
pyyaml
jax
jaxlib
numba
tensorflow
intersect
""".strip())

    # Install requirements and the package
    commands = [
        f"{activate_script} && pip install -r requirements.txt",
        f"{activate_script} && pip install -e ."
    ]

    for cmd in commands:
        run_command(cmd)

    print("Environment setup complete!")

if __name__ == "__main__":
    main()