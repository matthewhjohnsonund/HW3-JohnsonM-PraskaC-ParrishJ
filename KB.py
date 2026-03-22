"""Representations and Inference for Logic (Chapters 7-10)

Covers both Propositional and First-Order Logic. First we have four
important data types:

    KB            Abstract class holds a knowledge base of logical expressions
    KB_Agent      Abstract class subclasses agents.Agent
    Expr          A logical expression, imported from utils.py
    substitution  Implemented as a dictionary of var:value pairs, {x:1, y:x}

Be careful: some functions take an Expr as argument, and some take a KB.

Logical expressions can be created with Expr or expr, imported from utils, TODO
or with expr, which adds the capability to write a string that uses
the connectives ==>, <==, <=>, or <=/=>. But be careful: these have the
operator precedence of commas; you may need to add parents to make precedence work.
See logic.ipynb for examples.

Then we implement various functions for doing logical inference:

    pl_true          Evaluate a propositional logical sentence in a model
    tt_entails       Say if a statement is entailed by a KB
    pl_resolution    Do resolution on propositional sentences
    dpll_satisfiable See if a propositional sentence is satisfiable
    WalkSAT          Try to find a solution for a set of clauses

And a few other functions:

    to_cnf           Convert to conjunctive normal form
    unify            Do unification of two FOL sentences
    diff, simp       Symbolic differentiation and simplification
"""
import itertools
import random
from collections import defaultdict

from logic import (Expr, expr, subexpressions, to_cnf, conjuncts, disjuncts, pl_resolution, pl_fc_entails, is_symbol, is_prop_symbol, is_definite_clause, parse_definite_clause, first)

class Glitter:  pass
class Bump:     pass
class Stench:   pass
class Breeze:   pass
class Scream:   pass


# ______________________________________________________________________________
# Chapter 7 Logical Agents
# 7.1 Knowledge Based Agents


class KB:
    """
    A knowledge base to which you can tell and ask sentences.
    To create a KB, subclass this class and implement tell, ask_generator, and retract.
    Ask_generator:
      For a Propositional Logic KB, ask(P & Q) returns True or False, but for an
      FOL KB, something like ask(Brother(x, y)) might return many substitutions
      such as {x: Cain, y: Abel}, {x: Abel, y: Cain}, {x: George, y: Jeb}, etc.
      So ask_generator generates these one at a time, and ask either returns the
      first one or returns False.
    """

    def __init__(self, sentence=None):
        raise NotImplementedError

    def tell(self, sentence):
        """Add the sentence to the KB."""
        raise NotImplementedError

    def ask(self, query):
        """Return a substitution that makes the query true, or, failing that, return False."""
        return first(self.ask_generator(query), default=False)

    def ask_generator(self, query):
        """Yield all the substitutions that make query true."""
        raise NotImplementedError

    def retract(self, sentence):
        """Remove sentence from the KB."""
        raise NotImplementedError
