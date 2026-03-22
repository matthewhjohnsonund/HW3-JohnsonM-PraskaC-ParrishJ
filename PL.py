from logic import (is_symbol, is_var_symbol, is_prop_symbol, subexpressions, is_definite_clause, parse_definite_clause, Expr, expr, first)


def is_symbol(s):
    """A string s is a symbol if it starts with an alphabetic char.
    >>> is_symbol('R2D2')
    True
    """
    return isinstance(s, str) and s[:1].isalpha()


def is_var_symbol(s):
    """A logic variable symbol is an initial-lowercase string.
    >>> is_var_symbol('EXE')
    False
    """
    return is_symbol(s) and s[0].islower()


def is_prop_symbol(s):
    """A proposition logic symbol is an initial-uppercase string.
    >>> is_prop_symbol('exe')
    False
    """
    return is_symbol(s) and s[0].isupper()


def variables(s):
    """Return a set of the variables in expression s.
    >>> variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, 2)')) 
    == {x, y, z}
    True	
    """
    return {x for x in subexpressions(s) if is_var_symbol(x.op)}


def is_definite_clause(s):
    """
    Returns True for exprs s of the form A & B & ... & C ==> D,
    where all literals are positive.  In clause form, this is
    ~A | ~B | ... | ~C | D, where exactly one clause is positive.
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    """
    from logic import conjuncts as _conjuncts
    if is_symbol(s.op):
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return (is_symbol(consequent.op) and
                all(is_symbol(arg.op) for arg in _conjuncts(antecedent)))
    else:
        return False


def parse_definite_clause(s):
    """Return the antecedents and the consequent of a definite clause."""
    from logic import conjuncts as _conjuncts
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return _conjuncts(antecedent), consequent


# Useful constant Exprs used in examples and code:
A, B, C, D, E, F, G, P, Q, x, y, z = map(Expr, 'ABCDEFGPQxyz')
