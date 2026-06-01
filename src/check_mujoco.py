import mujoco
import numpy as np
import platform
import sys




def main() -> None:
    print("Python version:", sys.version)
    print("Operating system:", platform.platform())
    print("MuJoCo version:", mujoco.__version__)
    print("NumPy version:", np.__version__)
    print("Environment check passed.")


if __name__ == "__main__":
    main()

