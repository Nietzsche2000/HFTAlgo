import pandas as pd

# STEP 1: GENERATE PRIME NUMBERS
def is_prime(num):
    """Check if a number is prime."""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def generate_primes(n):
    """Generate a list of the first 'n' prime numbers."""
    primes = []
    i = 2  # Starting from the first prime number
    while len(primes) < n:
        if is_prime(i):
            primes.append(i)
        i += 1
    return primes

# STEP 2: CREATE A LIST OF PRIMES
n = 100  # Specify how many prime numbers you want
primes = generate_primes(n)

# STEP 3: CREATE A DATAFRAME
# Create a DataFrame with corresponding indices
df = pd.DataFrame({
    'Index': range(1, n + 1),
    'PrimeNumber': primes
})

# STEP 4: EXPORT TO CSV
csv_file = 'prime_numbers.csv'
df.to_csv(csv_file, index=False)

print(f'Prime numbers CSV created: {csv_file}')
