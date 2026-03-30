# 7.5.4 Forward and backward chaining

# Extended – removed circular pl_fc_entails import
from collections import defaultdict
from KB import KB, Expr, expr, first, is_prop_symbol, is_definite_clause, conjuncts
from PropKB import PropKB

class PropDefiniteKB(PropKB):
    """A KB of propositional definite clauses."""

    def tell(self, sentence):
        """Add a definite clause to this KB."""
        assert is_definite_clause(sentence), "Must be definite clause"
        self.clauses.append(sentence)

    def ask_generator(self, query):
        """Yield the empty substitution if KB implies query; else nothing."""
        if pl_fc_entails(self.clauses, query):
            yield {}

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def clauses_with_premise(self, p):
        """Return a list of the clauses in KB that have p in their premise.
        This could be cached away for O(1) speed, but we'll recompute it."""
        return [c for c in self.clauses
                if c.op == '==>' and p in conjuncts(c.args[0])]


def pl_fc_entails(KB, q):
    """Use forward chaining to see if a PropDefiniteKB entails symbol q.
    [Figure 7.15]
    >>> pl_fc_entails(horn_clauses_KB, expr('Q'))
    True
    """
    count = {c: len(conjuncts(c.args[0]))
             for c in KB.clauses
             if c.op == '==>'}
    inferred = defaultdict(bool)
    agenda = [s for s in KB.clauses if is_prop_symbol(s.op)]
    while agenda:
        p = agenda.pop()
        if p == q:
            return True
        if not inferred[p]:
            inferred[p] = True
            for c in KB.clauses_with_premise(p):
                count[c] -= 1
                if count[c] == 0:
                    agenda.append(c.args[1])
    return False


""" [Figure 7.13]
Simple inference in a wumpus world example
"""
wumpus_world_inference = expr("(B11 <=> (P12 | P21))  &  ~B11")

""" [Figure 7.16]
Propositional Logic Forward Chaining example
"""
horn_clauses_KB = PropDefiniteKB()
for s in "P==>Q; (L&M)==>P; (B&L)==>M; (A&P)==>L; (A&B)==>L; A;B".split(';'):
    horn_clauses_KB.tell(expr(s))

"""
Definite clauses KB example
"""
definite_clauses_KB = PropDefiniteKB()
for clause in ['(B & F)==>E', '(A & E & F)==>G', '(B & C)==>F', '(A & B)==>D', '(E & F)==>H', '(H & I)==>J', 'A', 'B',
               'C']:
    definite_clauses_KB.tell(expr(clause))
