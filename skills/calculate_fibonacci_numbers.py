def calculate_fibonacci_numbers() -> str:
    """
    Calculate the first 10 Fibonacci numbers using recursion.

    Returns:
        A string describing the result of the execution.
    """

    def fibonacci(n: int) -> int:
        """Calculate the nth Fibonacci number."""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fibonacci(n - 1) + fibonacci(n - 2)

    try:
        # Calculate the first 10 Fibonacci numbers
        fib_numbers = [fibonacci(i) for i in range(10)]
        
        # Return a string describing the result
        return f"The first 10 Fibonacci numbers are: {fib_numbers}"
    
    except Exception as e:
        # Return an error message if something goes wrong
        return f"An error occurred: {str(e)}"