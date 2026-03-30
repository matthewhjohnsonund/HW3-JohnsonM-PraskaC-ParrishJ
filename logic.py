"""
logic.py
Extended - logic expressions and inference algorithms for propositional logic
"""

import itertools
from collections import defaultdict

class Expr:
    """A mathematical expression with an op (operator) and 0 or more args"""

    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args

    def __neg__(self):       return Expr('~', self)
    def __invert__(self):    return Expr('~', self)
    def __and__(self, other):  return Expr('&', self, other)
    def __or__(self, other):   return Expr('|', self, other)
    def __rshift__(self, other): return Expr('==>', self, other)
    def __lshift__(self, other): return Expr('<==', self, other)
    def __xor__(self, other):  return Expr('^', self, other)
    def __eq__(self, other):
        return isinstance(other, Expr) and self.op == other.op and self.args == other.args
    def __hash__(self):
        return hash(self.op) ^ hash(self.args)
    def __repr__(self):
        if not self.args:
            return str(self.op)
        elif len(self.args) == 1:
            return '{}({})'.format(self.op, self.args[0])
        else:
            return '{}({})'.format(self.op, ', '.join(map(str, self.args)))
    def __call__(self, *args):
        """Allow Expr('F')(x, y) as a convenience"""
        return Expr(self.op, *args)


def expr(x):
    """Create an Expr from a string using simple parsing"""
    if isinstance(x, str):
        return expr_parse(x.strip())
    return x


def expr_parse(s):
    """Very small recursive-descent parser for propositional logic strings"""
    s = s.strip()

    for op in ['<=>', '<=>']:
        idx = find_main_connective(s, op)
        if idx >= 0:
            return Expr('<=>', expr_parse(s[:idx]), expr_parse(s[idx+3:]))

    idx = find_main_connective(s, '==>')
    if idx >= 0:
        return Expr('==>', expr_parse(s[:idx]), expr_parse(s[idx+3:]))

    idx = find_main_connective(s, '|')
    if idx >= 0:
        return Expr('|', expr_parse(s[:idx]), expr_parse(s[idx+1:]))

    idx = find_main_connective(s, '&')
    if idx >= 0:
        return Expr('&', expr_parse(s[:idx]), expr_parse(s[idx+1:]))

    if s.startswith('~') or s.startswith('!'):
        return Expr('~', expr_parse(s[1:]))

    if s.startswith('(') and s.endswith(')') and matching_paren(s):
        return expr_parse(s[1:-1])

    if '(' in s:
        name = s[:s.index('(')]
        inner = s[s.index('(')+1:-1]
        args = split_args(inner)
        return Expr(name, *[expr_parse(a) for a in args])

    return Expr(s)


def find_main_connective(s, op):
    """Find the index of op at the top level (depth 0) in string s"""
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0 and s[i:i+len(op)] == op:
            return i
        i += 1
    return -1


def matching_paren(s):
    """Return True if the first '(' matches the last ')'"""
    depth = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0 and i < len(s) - 1:
                return False
    return depth == 0


def split_args(s):
    """Split comma-separated args respecting parentheses"""
    args, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(s[start:i].strip())
            start = i + 1
    args.append(s[start:].strip())
    return args


def subexpressions(x):
    """Yield all sub-expressions of x"""
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpressions(arg)

def to_cnf(s):
    """Convert a propositional sentence to CNF"""
    if isinstance(s, str):
        s = expr(s)
    s = eliminate_implications(s)
    s = move_not_inwards(s)
    s = distribute_and_over_or(s)
    return s


def eliminate_implications(s):
    if not s.args or is_symbol(s.op):
        return s
    args = list(map(eliminate_implications, s.args))
    a, b = args[0], args[-1]
    if s.op == '==>':
        return b | ~a
    elif s.op == '<==':
        return a | ~b
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)


def move_not_inwards(s):
    if s.op == '~':
        def NOT(b): return move_not_inwards(~b)
        a = s.args[0]
        if a.op == '~':
            return move_not_inwards(a.args[0])
        if a.op == '&':
            return associate('|', list(map(NOT, a.args)))
        if a.op == '|':
            return associate('&', list(map(NOT, a.args)))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *list(map(move_not_inwards, s.args)))


def distribute_and_over_or(s):
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        args = list(map(distribute_and_over_or, s.args))
        s = associate('|', args)
        if s.op != '|':
            return s
        
        conj = first(a for a in s.args if a.op == '&')
        if not conj:
            return s
        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c | rest) for c in conj.args])
    elif s.op == '&':
        return associate('&', list(map(distribute_and_over_or, s.args)))
    else:
        return s


def associate(op, args):
    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)


_op_identity = {'&': Expr('T'), '|': Expr('F'), '+': Expr('0'), '*': Expr('1')}


def dissociate(op, args):
    result = []
    def collect(subargs):
        for arg in subargs:
            if isinstance(arg, Expr) and arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)
    collect(args)
    return result


def conjuncts(s):
    return dissociate('&', [s])


def disjuncts(s):
    return dissociate('|', [s])

def is_symbol(s):
    return isinstance(s, str) and s[:1].isalpha()


def is_prop_symbol(s):
    return is_symbol(s) and s[0].isupper()


def is_var_symbol(s):
    return is_symbol(s) and s[0].islower()


def is_definite_clause(s):
    if is_symbol(s.op):
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return (is_symbol(consequent.op) and
                all(is_symbol(arg.op) for arg in conjuncts(antecedent)))
    else:
        return False


def parse_definite_clause(s):
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent

def pl_fc_entails(kb_clauses, q):
    """Forward chaining for PropDefiniteKB."""
    count = {c: len(conjuncts(c.args[0])) for c in kb_clauses if c.op == '==>'}
    inferred = defaultdict(bool)
    agenda = [s for s in kb_clauses if is_prop_symbol(s.op)]
    while agenda:
        p = agenda.pop()
        if p == q:
            return True
        if not inferred[p]:
            inferred[p] = True
            for c in kb_clauses:
                if c.op == '==>' and p in conjuncts(c.args[0]):
                    count[c] -= 1
                    if count[c] == 0:
                        agenda.append(c.args[1])
    return False

def pl_resolution(kb, alpha):
    """Propositional resolution refutation."""
    clauses = kb.clauses + conjuncts(to_cnf(~alpha))
    new = set()
    while True:
        n = len(clauses)
        pairs = [(clauses[i], clauses[j])
                 for i in range(n) for j in range(i + 1, n)]
        for (ci, cj) in pairs:
            resolvents = pl_resolve(ci, cj)
            if Expr('F') in resolvents:
                return True
            new |= set(resolvents)
        if new.issubset(set(clauses)):
            return False
        for c in new:
            if c not in clauses:
                clauses.append(c)


def pl_resolve(ci, cj):
    clauses = []
    di = disjuncts(ci)
    dj = disjuncts(cj)
    for ai in di:
        for aj in dj:
            if ai == ~aj or ~ai == aj:
                new_d = [a for a in di if a != ai] + [a for a in dj if a != aj]
                clauses.append(associate('|', new_d))
    return clauses

def first(iterable, default=None):
    for item in iterable:
        return item
    return default


def remove_all(item, seq):
    return [x for x in seq if x != item]


def unique(seq):
    return list(dict.fromkeys(seq))
