# A module for financial calculations

# finance.py

def simple(rate, years, principal):
    return principal * (1 + rate * years)

def compound(rate, years, principal):
    return principal * (1 + rate) ** years

def amortize(rate, years, principal):
    return principal * rate / (1 - (1 + rate) ** -years)

def present(value, rate, years):
    return value / (1 + rate) ** years
