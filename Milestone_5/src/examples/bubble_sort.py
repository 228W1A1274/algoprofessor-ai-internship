# bubble_sort.py
# Example program for CodeXcelerate transpilation
# Algorithm: Bubble Sort — O(n²) — intentionally slow in Python
# Expected speedup after transpilation: 500–2000x

import random

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

random.seed(42)
data = [random.randint(0, 100000) for _ in range(10000)]

result = bubble_sort(data)
print(f"Sorted {len(result)} elements")
print(f"First 5: {result[:5]}")
print(f"Last 5:  {result[-5:]}")
