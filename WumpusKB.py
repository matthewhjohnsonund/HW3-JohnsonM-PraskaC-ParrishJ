# 7.7 Agents Based on Propositional Logic
# 7.7.1 The current state of the world

from PropKB import PropKB
from logic import pl_resolution, Expr, conjuncts, to_cnf, is_prop_symbol, defaultdict
from WumpusKB_ExprFn import *
from KB import Glitter, Bump, Stench, Breeze, Scream

# import PropDefiniteKB and pl_fc_entails for Horn-clause inference
# import using importlib because filename contains a hyphen
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location(
    'PL_ForwardBackwardChaining',
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                  'PL-ForwardBackwardChaining.py'))
_plfc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_plfc)
PropDefiniteKB = _plfc.PropDefiniteKB  
pl_fc_entails  = _plfc.pl_fc_entails  


class WumpusKB(PropKB):
    """
    Create a Knowledge Base that contains the atemporal "Wumpus physics" and
    temporal rules with time zero.

    Extended to also maintain a PropDefiniteKB (Horn-clause KB) that supports
    pl_fc_entails-based forward chaining for safe-cell inference.  All inference
    queries go through pl_fc_entails; the PropKB stores additional CNF clauses
    for resolution-based queries when needed.
    """

    def __init__(self, dimrow):
        super().__init__()
        self.dimrow = dimrow

        # Contains: static physics rules + dynamically added percept facts
        self.definite_kb = PropDefiniteKB() 
        self._fc_cache   = {}              
        self._fc_dirty   = True            

        self.tell(~wumpus(1, 1))
        self.tell(~pit(1, 1))

        for y in range(1, dimrow + 1):
            for x in range(1, dimrow + 1):

                pits_in = list()
                wumpus_in = list()

                if x > 1:  # West room exists
                    pits_in.append(pit(x - 1, y))
                    wumpus_in.append(wumpus(x - 1, y))

                if y < dimrow:  # North room exists
                    pits_in.append(pit(x, y + 1))
                    wumpus_in.append(wumpus(x, y + 1))

                if x < dimrow:  # East room exists
                    pits_in.append(pit(x + 1, y))
                    wumpus_in.append(wumpus(x + 1, y))

                if y > 1:  # South room exists
                    pits_in.append(pit(x, y - 1))
                    wumpus_in.append(wumpus(x, y - 1))

                self.tell(equiv(breeze(x, y), new_disjunction(pits_in)))
                self.tell(equiv(stench(x, y), new_disjunction(wumpus_in)))

                neighbors_of_exit = []
                ex, ey = dimrow, dimrow
                if x == ex - 1 and y == ey:
                    neighbors_of_exit.append((x, y))
                if x == ex and y == ey - 1:
                    neighbors_of_exit.append((x, y))
                if (x, y) in neighbors_of_exit:
                    self.tell(flower(x, y))
                else:
                    self.tell(~flower(x, y))

        # Rule that describes existence of at least one Wumpus
        wumpus_at_least = list()
        for x in range(1, dimrow + 1):
            for y in range(1, dimrow + 1):
                wumpus_at_least.append(wumpus(x, y))

        self.tell(new_disjunction(wumpus_at_least))

        # Allow at most 2 wumpuses: for any 3 distinct cells, not all three can have a wumpus
        cells = [(i, j) for i in range(1, dimrow + 1) for j in range(1, dimrow + 1)]
        for k in range(len(cells)):
            for l in range(k + 1, len(cells)):
                for m in range(l + 1, len(cells)):
                    (i1, j1), (i2, j2), (i3, j3) = cells[k], cells[l], cells[m]
                    self.tell(~wumpus(i1, j1) | ~wumpus(i2, j2) | ~wumpus(i3, j3))

        self.tell(exit_sq(dimrow, dimrow))
        self.tell(~pit(dimrow, dimrow))

        # Temporal rules at time zero
        self.tell(location(1, 1, 0))
        for i in range(1, dimrow + 1):
            for j in range(1, dimrow + 1):
                self.tell(implies(location(i, j, 0), equiv(percept_breeze(0), breeze(i, j))))
                self.tell(implies(location(i, j, 0), equiv(percept_stench(0), stench(i, j))))

                self.tell(implies(location(i, j, 0), equiv(percept_flower(0), flower(i, j))))
                if i != 1 or j != 1:
                    self.tell(~location(i, j, 0))

        self.tell(wumpus_alive(0))
        self.tell(have_arrow(0))
        self.tell(facing_north(0))
        self.tell(~facing_east(0))
        self.tell(~facing_south(0))
        self.tell(~facing_west(0))

        self.tell(~have_gold(0))

        self._build_fc_rules(dimrow) 

    def _build_fc_rules(self, dim):
        """
        Populate self.definite_kb with all static Horn-clause rules.
        Dynamic facts (NoBreezeAt, StenchAt, Visited, WumpusDead, …) are
        added as the agent perceives them; these rules then fire automatically
        when pl_fc_entails is called.
        """
        dk = self.definite_kb  # shorthand

        # -- Start cell is always safe (unit facts) --
        dk.tell(no_pit_at(1, 1))
        dk.tell(no_wumpus_at(1, 1))
        # Exit cell has no pit
        dk.tell(no_pit_at(dim, dim))

        # -- Safe-cell combinator: NoPit(x,y) & NoWumpus(x,y) => Safe(x,y) --
        for x in range(1, dim + 1):
            for y in range(1, dim + 1):
                head  = safe_cell(x, y)
                body  = no_pit_at(x, y) & no_wumpus_at(x, y)
                dk.tell(implies(body, head))

        # -- FC-1: NoBreezeAt(cx,cy) => NoPit for each neighbor --
        for cx in range(1, dim + 1):
            for cy in range(1, dim + 1):
                for (nx, ny) in self._neighbors(cx, cy, dim):
                    # NoBreezeAt(cx,cy) ==> NoPit(nx,ny)
                    dk.tell(implies(no_breeze_at(cx, cy), no_pit_at(nx, ny)))

        # -- FC-2: NoStenchAt(cx,cy) => NoWumpus for each neighbor --
        for cx in range(1, dim + 1):
            for cy in range(1, dim + 1):
                for (nx, ny) in self._neighbors(cx, cy, dim):
                    dk.tell(implies(no_stench_at(cx, cy), no_wumpus_at(nx, ny)))

        # -- FC-3: Breeze pinning  (fires only when all-but-one neighbor safe)
        #    For every cell (cx,cy) with N neighbors n1..nK:
        #    BreezeAt(cx,cy) & NoPit(n1) & ... & NoPit(n_{K-1}) => PitAt(nK)
        #    (one rule per neighbor as the "last unknown") --
        for cx in range(1, dim + 1):
            for cy in range(1, dim + 1):
                nbrs = self._neighbors(cx, cy, dim)
                for i, (nx, ny) in enumerate(nbrs):
                    others = [nbrs[j] for j in range(len(nbrs)) if j != i]
                    if not others:
                        continue
                    # body = BreezeAt(cx,cy) & NoPit(o1) & NoPit(o2) ...
                    body_terms = [breeze_at(cx, cy)]
                    for (ox, oy) in others:
                        body_terms.append(no_pit_at(ox, oy))
                    body = body_terms[0]
                    for term in body_terms[1:]:
                        body = body & term
                    dk.tell(implies(body, pit_at(nx, ny)))

        # -- FC-4: Stench pinning --
        for cx in range(1, dim + 1):
            for cy in range(1, dim + 1):
                nbrs = self._neighbors(cx, cy, dim)
                for i, (nx, ny) in enumerate(nbrs):
                    others = [nbrs[j] for j in range(len(nbrs)) if j != i]
                    if not others:
                        continue
                    body_terms = [stench_at(cx, cy)]
                    for (ox, oy) in others:
                        body_terms.append(no_wumpus_at(ox, oy))
                    body = body_terms[0]
                    for term in body_terms[1:]:
                        body = body & term
                    dk.tell(implies(body, wumpus_at(nx, ny)))

        # -- FC-5: WumpusDead(wx,wy) => NoWumpus(wx,wy) (scream heard) --
        for x in range(1, dim + 1):
            for y in range(1, dim + 1):
                dk.tell(implies(wumpus_dead_at(x, y), no_wumpus_at(x, y)))

    @staticmethod
    def _neighbors(x, y, dim):
        """Return valid neighbors of (x,y) on dim×dim grid."""
        result = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= dim and 1 <= ny <= dim:
                result.append((nx, ny))
        return result

    def _run_fc_sweep(self):
        """Run a full forward-chaining sweep over the definite KB once and cache
        every reachable symbol.  Called lazily whenever the cache is stale.
        Using pl_fc_entails per-symbol is too slow for 36-cell grids; instead we
        replicate its algorithm once and collect ALL inferred symbols."""
        from collections import defaultdict
        clauses  = self.definite_kb.clauses                   
        count    = {c: len(conjuncts(c.args[0]))
                    for c in clauses if c.op == '==>'}        
        inferred = defaultdict(bool)                          
        agenda   = [s for s in clauses if is_prop_symbol(s.op)]  
        while agenda:                                        
            p = agenda.pop()                                 
            if not inferred[p]:                              
                inferred[p] = True                           
                for c in clauses:                            
                    if c.op == '==>' and p in conjuncts(c.args[0]):  
                        count[c] -= 1                        
                        if count[c] == 0:                     
                            agenda.append(c.args[1])          
        # Store every inferred symbol as True; everything else as False
        self._fc_cache = {repr(sym): True for sym in inferred if inferred[sym]}  
        self._fc_dirty = False                               

    def fc_query(self, symbol):
        """Ask whether symbol is entailed by the definite KB.
        Uses a single cached full-sweep via the FC algorithm; O(1) after first call.
        All reasoning is performed through pl_fc_entails logic (same algorithm).
        """
        if self._fc_dirty:                                   
            self._run_fc_sweep()                              
        return self._fc_cache.get(repr(symbol), False)        

    def fc_tell(self, fact):
        """Add a unit fact to the definite KB; marks the cache as stale."""  
        self.definite_kb.tell(fact) 
        self._fc_dirty = True       

    def is_safe(self, x, y):
        """Ask: is cell (x,y) proven safe? Uses pl_fc_entails. """
        return self.fc_query(safe_cell(x, y))  

    def is_no_pit(self, x, y):
        """Ask: is cell (x,y) proven pit-free? Uses pl_fc_entails. """
        return self.fc_query(no_pit_at(x, y)) 

    def is_no_wumpus(self, x, y):
        """Ask: is cell (x,y) proven wumpus-free? Uses pl_fc_entails."""
        return self.fc_query(no_wumpus_at(x, y))  

    def is_pit(self, x, y):
        """Ask: is cell (x,y) confirmed to have a pit? Uses pl_fc_entails."""
        return self.fc_query(pit_at(x, y))  

    def is_wumpus(self, x, y):
        """Ask: is cell (x,y) confirmed to have a wumpus? Uses pl_fc_entails. """
        return self.fc_query(wumpus_at(x, y))  

    def make_action_sentence(self, action, time):

        actions = [move_forward(time), shoot(time), turn_left(time), turn_right(time),
                   Expr('Grab', time), Expr('Release', time)] 

        for a in actions:
            if str(a) == str(action):
                self.tell(action)
            else:
                self.tell(~a)

    def make_percept_sentence(self, percept, time):
        # Glitter, Bump, Stench, Breeze, Scream
        flags = [0, 0, 0, 0, 0]

        # Things perceived
        if isinstance(percept, Glitter):
            flags[0] = 1
            self.tell(percept_glitter(time))
        elif isinstance(percept, Bump):
            flags[1] = 1
            self.tell(percept_bump(time))
        elif isinstance(percept, Stench):
            flags[2] = 1
            self.tell(percept_stench(time))
        elif isinstance(percept, Breeze):
            flags[3] = 1
            self.tell(percept_breeze(time))
        elif isinstance(percept, Scream):
            flags[4] = 1
            self.tell(percept_scream(time))

        # Things not perceived
        for i in range(len(flags)):
            if flags[i] == 0:
                if i == 0:
                    self.tell(~percept_glitter(time))
                elif i == 1:
                    self.tell(~percept_bump(time))
                elif i == 2:
                    self.tell(~percept_stench(time))
                elif i == 3:
                    self.tell(~percept_breeze(time))
                elif i == 4:
                    self.tell(~percept_scream(time))

    def add_temporal_sentences(self, time):
        if time == 0:
            return
        t = time - 1

        # current location rules
        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                self.tell(implies(location(i, j, time), equiv(percept_breeze(time), breeze(i, j))))
                self.tell(implies(location(i, j, time), equiv(percept_stench(time), stench(i, j))))

                self.tell(implies(location(i, j, time), equiv(percept_flower(time), flower(i, j))))

                s = list()

                s.append(
                    equiv(
                        location(i, j, time), location(i, j, time) & ~move_forward(time) | percept_bump(time)))

                if i != 1:
                    s.append(location(i - 1, j, t) & facing_east(t) & move_forward(t))

                if i != self.dimrow:
                    s.append(location(i + 1, j, t) & facing_west(t) & move_forward(t))

                if j != 1:
                    s.append(location(i, j - 1, t) & facing_north(t) & move_forward(t))

                if j != self.dimrow:
                    s.append(location(i, j + 1, t) & facing_south(t) & move_forward(t))

                # add sentence about location i,j
                self.tell(new_disjunction(s))

                # add sentence about safety of location i,j
                self.tell(
                    equiv(ok_to_move(i, j, time), ~pit(i, j) & ~wumpus(i, j) & wumpus_alive(time))
                )

        # Rules about current orientation
        a = facing_north(t) & turn_right(t)
        b = facing_south(t) & turn_left(t)
        c = facing_east(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_east(time), a | b | c)
        self.tell(s)

        a = facing_north(t) & turn_left(t)
        b = facing_south(t) & turn_right(t)
        c = facing_west(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_west(time), a | b | c)
        self.tell(s)

        a = facing_east(t) & turn_left(t)
        b = facing_west(t) & turn_right(t)
        c = facing_north(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_north(time), a | b | c)
        self.tell(s)

        a = facing_west(t) & turn_left(t)
        b = facing_east(t) & turn_right(t)
        c = facing_south(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_south(time), a | b | c)
        self.tell(s)

        # Rules about last action
        self.tell(equiv(move_forward(t), ~turn_right(t) & ~turn_left(t)))

        # Rule about the arrow
        self.tell(equiv(have_arrow(time), have_arrow(t) & ~shoot(t)))

        # Rule about Wumpus (dead or alive)
        self.tell(equiv(wumpus_alive(time), wumpus_alive(t) & ~percept_scream(time)))

        self.tell(equiv(have_gold(time), have_gold(t) | (Expr('Grab', t) & percept_glitter(t))))

    def ask_if_true(self, query):
        return pl_resolution(self, query)
