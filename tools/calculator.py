def calculate_percentage(obtained: float, total: float) -> float:
    """
    Calculate percentage from obtained marks and total marks.
    """

    if total <= 0:
        raise ValueError("Total marks must be greater than zero.")

    return (obtained / total) * 100


def calculate_average(values: list[float]) -> float:
    """
    Calculate the average of a list of numbers.
    """

    if not values:
        raise ValueError("At least one value is required.")

    return sum(values) / len(values)