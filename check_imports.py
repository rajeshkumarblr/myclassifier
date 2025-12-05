
import time
print("Importing torch...")
start = time.time()
import torch
print(f"Torch imported in {time.time() - start:.2f}s")

print("Importing torchvision...")
start = time.time()
import torchvision
print(f"Torchvision imported in {time.time() - start:.2f}s")

print("Importing sympy...")
start = time.time()
import sympy
print(f"Sympy imported in {time.time() - start:.2f}s")
