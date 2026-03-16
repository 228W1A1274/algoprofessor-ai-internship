# prime_sieve.py
# Example program for CodeXcelerate transpilation
# Algorithm: Sieve of Eratosthenes — O(n log log n)
# Expected speedup after transpilation: 50–200x

def sieve_of_eratosthenes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    i = 2
    while i * i <= limit:
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
        i += 1

    return [num for num in range(2, limit + 1) if is_prime[num]]

LIMIT = 5_000_000
primes = sieve_of_eratosthenes(LIMIT)
print(f"Found {len(primes)} primes up to {LIMIT:,}")
print(f"Largest prime: {primes[-1]}")
print(f"First 10 primes: {primes[:10]}")
