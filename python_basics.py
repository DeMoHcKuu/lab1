"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(character.lower() in "aeiou" for character in text)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return number.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Vectors must have the same length")
    if not predicted:
        return 0.0
    return sum((actual - target) ** 2 for actual, target in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors: list[str] = []
    divisor = 2
    remaining = number

    while divisor * divisor <= remaining:
        power = 0
        while remaining % divisor == 0:
            remaining //= divisor
            power += 1
        if power:
            factors.append(f"({divisor}{f'**{power}' if power > 1 else ''})")
        divisor = 3 if divisor == 2 else divisor + 2

    if remaining > 1:
        factors.append(f"({remaining})")
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    layer = 1
    total = 0
    while total < cube_count:
        total += layer * layer
        if total == cube_count:
            return layer
        layer += 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    middle = len(digits) // 2
    if len(digits) % 2:
        left, right = digits[:middle], digits[middle + 1 :]
    else:
        left, right = digits[: middle - 1], digits[middle + 1 :]
    return sum(map(int, left)) == sum(map(int, right))
