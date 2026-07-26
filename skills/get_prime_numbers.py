def get_prime_numbers() -> str:
    """
    This function generates and prints the first 20 prime numbers using a loop to check each number for primality.

    Returns:
        str: A message indicating the result of the execution.
    """

    def is_prime(n: int) -> bool:
        """Check if a number is prime."""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    prime_numbers = []
    num = 2
    try:
        while len(prime_numbers) < 20:
            if is_prime(num):
                prime_numbers.append(num)
            num += 1
        result = "The first 20 prime numbers are: " + ", ".join(map(str, prime_numbers))
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"