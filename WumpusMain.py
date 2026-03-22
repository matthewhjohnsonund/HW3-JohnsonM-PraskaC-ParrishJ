"""
WumpusMain.py

DATE: 3-31-26

STUDENT NAME: Matthew Johnson
STUDENT ID: 1394331

STUDENT NAME: Jack Parrish
STUDENT ID: 1281351

STUDENT NAME: Connor Praska
STUDENT ID: 1347238

PEAS:
  Environment : 6x6 grid, start [1,1] facing north, exit [6,6]
  Gold        : one piece randomly placed, two Wumpuses randomly placed
  Pits        : each non-start/non-exit square has probability 0.2
  Sensors     : Stench, Breeze, Glitter, Flower (adj to exit), Bump, Scream
  Actuators   : Forward, TurnLeft, TurnRight, Grab, Release, Shoot, Climb
  Score       : +600 grab gold, +400 exit safely, -1000 death,
                -1 per step, -10 shoot arrow

Inference: ALL reasoning is performed through WumpusKB(PropKB)

"""

import random                                
from collections import deque               

from WumpusKB import WumpusKB            
from WumpusKB_ExprFn import (             
    no_breeze_at, breeze_at,
    no_stench_at, stench_at,
    no_pit_at, pit_at,
    no_wumpus_at, wumpus_at,
    safe_cell, visited_cell,
    wumpus_dead_at, have_gold_now,
    wumpus, pit, wumpus_alive, have_arrow,
    percept_stench, percept_breeze,
    percept_glitter, percept_bump, percept_scream, percept_flower,
    move_forward, shoot, turn_left, turn_right,
    ok_to_move, location, flower, exit_sq, gold, have_gold,
    facing_east, facing_west, facing_north, facing_south,
    implies, equiv, new_disjunction,
)
from WumpusPosition import WumpusPosition   

NORTH, EAST, SOUTH, WEST = 'N', 'E', 'S', 'W'
TURN_LEFT  = 'TurnLeft'
TURN_RIGHT = 'TurnRight'
FORWARD    = 'Forward'
GRAB       = 'Grab'
RELEASE    = 'Release'
SHOOT_ACT  = 'Shoot'
CLIMB      = 'Climb'
DIMROW     = 6

SCORE_GOLD  =  600
SCORE_EXIT  =  400
SCORE_DEATH = -1000
SCORE_STEP  = -1
SCORE_ARROW = -10

_TURN_L = {NORTH: WEST, WEST: SOUTH, SOUTH: EAST, EAST: NORTH}
_TURN_R = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}
_FWD    = {NORTH: (0, 1), EAST: (1, 0), SOUTH: (0, -1), WEST: (-1, 0)}

def turn_left_dir(d):  return _TURN_L[d]                     
def turn_right_dir(d): return _TURN_R[d]                     

def forward_cell(x, y, d):                                   
    dx, dy = _FWD[d]
    return x + dx, y + dy

def steps_to_rotate(cur, des):                               
    """Minimal turn sequence to go from direction."""
    if cur == des:                 return []
    if turn_right_dir(cur) == des: return [TURN_RIGHT]
    if turn_left_dir(cur)  == des: return [TURN_LEFT]
    return [TURN_RIGHT, TURN_RIGHT]   # 180 turn

def grid_neighbors(x, y, dim=DIMROW):                        
    """Return valid adjacent cells on a dim×dim grid."""
    return [(x+dx, y+dy)
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
            if 1 <= x+dx <= dim and 1 <= y+dy <= dim]

def bfs_path(start, goal, passable):                         
    """BFS from start to goal restricted to the passable cell set.
    Returns list of cells (excluding start) or None if unreachable."""
    if start == goal:
        return []
    queue = deque([(start, [])])
    seen  = {start}
    while queue:
        (cx, cy), path = queue.popleft()
        for nxt in grid_neighbors(cx, cy):
            if nxt in passable and nxt not in seen:
                new_path = path + [nxt]
                if nxt == goal:
                    return new_path
                seen.add(nxt)
                queue.append((nxt, new_path))
    return None

def path_to_actions(path, sx, sy, orient):                   
    """Convert a cell-path into a sequence of TurnLeft/TurnRight/Forward."""
    actions = []
    cx, cy, d = sx, sy, orient
    for (nx, ny) in path:
        if   ny > cy: desired = NORTH
        elif ny < cy: desired = SOUTH
        elif nx > cx: desired = EAST
        else:         desired = WEST
        actions.extend(steps_to_rotate(d, desired))
        d = desired
        actions.append(FORWARD)
        cx, cy = nx, ny
    return actions

class WumpusWorldEnv:                                         

    def __init__(self, dimrow=DIMROW, seed=None):            
        self.dimrow = dimrow
        rng   = random.Random(seed)                        
        start = (1, 1)
        exit_ = (dimrow, dimrow)

        #    Place pits with p=0.2 on every non-start/non-exit cell   
        self.pits = set()                                   
        for x in range(1, dimrow + 1):
            for y in range(1, dimrow + 1):
                if (x, y) in (start, exit_):
                    continue
                if rng.random() < 0.2:                      
                    self.pits.add((x, y))

        #    Place exactly 2 Wumpuses in pit-free non-start/non-exit cells   
        wumpus_candidates = [                               
            (x, y)
            for x in range(1, dimrow + 1)
            for y in range(1, dimrow + 1)
            if (x, y) not in (start, exit_) and (x, y) not in self.pits
        ]
        rng.shuffle(wumpus_candidates)                      
        self.wumpus_positions = wumpus_candidates[:2]       
        self._live_wumpus     = set(self.wumpus_positions)  

        #    Place 1 gold in a remaining valid cell   
        gold_candidates = [                                 
            c for c in wumpus_candidates[2:]
            if c not in (start, exit_)
        ]
        self.gold_pos = rng.choice(gold_candidates) if gold_candidates else (2, 2)   

    def has_pit(self, x, y):            return (x, y) in self.pits                  
    def has_wumpus(self, x, y):         return (x, y) in set(self.wumpus_positions)  
    def has_live_wumpus(self, x, y):    return (x, y) in self._live_wumpus           
    def kill_wumpus(self, x, y):        self._live_wumpus.discard((x, y))            

    def percepts_at(self, x, y):                            
        """Return dict of boolean percepts at cell (x,y)."""
        dim  = self.dimrow
        nbrs = grid_neighbors(x, y, dim)
        return {
            'breeze':  any(self.has_pit(nx, ny) for nx, ny in nbrs),          
            'stench':  any(self.has_live_wumpus(nx, ny) for nx, ny in nbrs),  
            'glitter': (x, y) == self.gold_pos,                                
            'flower':  (x, y) in [(dim-1, dim), (dim, dim-1)],                
        }


class HybridWumpusAgent:                                      
    """
    Knowledge-based Wumpus World agent.
    Decision-making is driven entirely by WumpusKB(PropKB) queried through
    pl_fc_entails.  No custom inference sets are maintained outside the KB.
     
    """

    def __init__(self, dimrow=DIMROW, world=None):           
        self.dimrow       = dimrow
        self.world        = world
        self.kb           = WumpusKB(dimrow)                
        self.pos          = WumpusPosition(1, 1, NORTH)     
        self.time         = 0
        self.score        = 0
        self.actions_log  = []                              
        self.have_gold    = False
        self.have_arrow   = True
        self.alive        = True
        self.exited       = False
        self.wumpus_dead  = set()                           

        self.visited          = {(1, 1)}                   
        self.new_facts_log    = []                       
        self.backtracks       = 0

        self._action_queue    = deque()                 
        self._plan_attempts   = set()                     

        # Tell KB that (1,1) is visited and safe from the start
        self.kb.fc_tell(no_pit_at(1, 1))                  
        self.kb.fc_tell(no_wumpus_at(1, 1))               
        self.kb.fc_tell(visited_cell(1, 1))               

    def run(self):                                           
        """Run the agent until it exits, dies, or hits the step limit."""
        print("\nWumpus World Starting")
        print("Grid: {}x{}, Start: [1,1] North, Exit: [{},{}]"
              .format(self.dimrow, self.dimrow, self.dimrow, self.dimrow))

        while self.alive and not self.exited and self.time < 500:
            cx, cy = self.pos.get_location()
            orient = self.pos.get_orientation()

            # Execute pre-planned action if queued
            if self._action_queue:
                self._execute(self._action_queue.popleft())
                continue

            #      Sense 
            percepts   = self.world.percepts_at(cx, cy)
            has_breeze = percepts['breeze']
            has_stench = percepts['stench']
            has_glitter= percepts['glitter']

            print("\n   Time {} | [{},{}] facing {}   "
                  .format(self.time, cx, cy, orient))
            print("  Percepts: breeze={} stench={} glitter={} flower={}"
                  .format(has_breeze, has_stench, has_glitter, percepts['flower']))

            #      Tell percepts to WumpusKB (PropKB + definite KB)     
            self._tell_percepts(cx, cy, has_breeze, has_stench,
                                has_glitter, percepts['flower'])

            #      Query KB via pl_fc_entails; log new inferences           
            self._query_and_log(cx, cy)

            #      Grab gold if glittering                                                           
            if has_glitter and not self.have_gold:
                self._execute(GRAB)
                continue

            #      Climb if at exit with gold                                                     
            if cx == self.dimrow and cy == self.dimrow and self.have_gold:
                self._execute(CLIMB)
                break

            #      Plan and queue next move                                                         
            self._plan_next(cx, cy, orient)

        if not self.alive:
            pass
        elif not self.exited:
            print("  [Agent halted — no safe path to goal]")

        self._print_results()
        return self.actions_log, self.score, self.backtracks


    def _tell_percepts(self, cx, cy, has_breeze, has_stench,   
                       has_glitter, has_flower):
        """Feed current percepts into WumpusKB as definite-clause facts.
        Tells both the Horn-clause definite KB (for pl_fc_entails inference)
        and the PropKB temporal layer (for resolution-based queries).
         """
        dk = self.kb                                        
        t  = self.time                                      

        #      Horn-clause KB: spatial percept facts (timeless, for FC)     
        if has_breeze:  dk.fc_tell(breeze_at(cx, cy))      
        else:           dk.fc_tell(no_breeze_at(cx, cy))   
        if has_stench:  dk.fc_tell(stench_at(cx, cy))      
        else:           dk.fc_tell(no_stench_at(cx, cy))   
        dk.fc_tell(visited_cell(cx, cy))                   

        #      PropKB temporal layer: percept symbols at time t                     
        if has_breeze:  dk.tell(percept_breeze(t))         
        else:           dk.tell(~percept_breeze(t))        
        if has_stench:  dk.tell(percept_stench(t))         
        else:           dk.tell(~percept_stench(t))        
        if has_glitter: dk.tell(percept_glitter(t))        
        else:           dk.tell(~percept_glitter(t))       
        if has_flower:  dk.tell(percept_flower(t))         
        else:           dk.tell(~percept_flower(t))        
        dk.tell(~percept_bump(t))                          
        dk.tell(~percept_scream(t))                        

 
    def _query_and_log(self, cx, cy):                       
        """
        Ask pl_fc_entails about all cells and log any facts newly entailed.
        This is the SOLE inference mechanism — no custom sets.
         
        """
        dim = self.dimrow
        for x in range(1, dim + 1):
            for y in range(1, dim + 1):
                # No-pit inference
                sym = no_pit_at(x, y)
                if self.kb.fc_query(sym):
                    key = "~P{},{}".format(x, y)
                    if key not in self._logged:
                        self._logged.add(key)
                        desc = "No Pit at [{},{}]".format(x, y)
                        print("  [KB] {} -- {}".format(key, desc))
                        self.new_facts_log.append((key, desc))

                # No-wumpus inference
                sym2 = no_wumpus_at(x, y)
                if self.kb.fc_query(sym2):
                    key2 = "~W{},{}".format(x, y)
                    if key2 not in self._logged:
                        self._logged.add(key2)
                        desc2 = "No Wumpus at [{},{}]".format(x, y)
                        print("  [KB] {} -- {}".format(key2, desc2))
                        self.new_facts_log.append((key2, desc2))

                # Safe-cell inference
                sym3 = safe_cell(x, y)
                if self.kb.fc_query(sym3):
                    key3 = "Safe{},{}".format(x, y)
                    if key3 not in self._logged:
                        self._logged.add(key3)
                        desc3 = "Safe cell [{},{}]".format(x, y)
                        print("  [KB] {} -- {}".format(key3, desc3))
                        self.new_facts_log.append((key3, desc3))

                # Pit inferred
                sym4 = pit_at(x, y)
                if self.kb.fc_query(sym4):
                    key4 = "P{},{}".format(x, y)
                    if key4 not in self._logged:
                        self._logged.add(key4)
                        desc4 = "Pit at [{},{}]".format(x, y)
                        print("  [KB] {} -- {}".format(key4, desc4))
                        self.new_facts_log.append((key4, desc4))

                # Wumpus inferred
                sym5 = wumpus_at(x, y)
                if self.kb.fc_query(sym5):
                    key5 = "W{},{}".format(x, y)
                    if key5 not in self._logged:
                        self._logged.add(key5)
                        desc5 = "Wumpus at [{},{}]".format(x, y)
                        print("  [KB] {} -- {}".format(key5, desc5))
                        self.new_facts_log.append((key5, desc5))

    # Override run() to initialise _logged set before loop           
    def run(self):                                           
        self._logged = set()                               
        print("\nWumpus World Starting")
        print("Grid: {}x{}, Start: [1,1] North, Exit: [{},{}]"
              .format(self.dimrow, self.dimrow, self.dimrow, self.dimrow))

        while self.alive and not self.exited and self.time < 500:
            cx, cy = self.pos.get_location()
            orient = self.pos.get_orientation()

            if self._action_queue:
                self._execute(self._action_queue.popleft())
                continue

            percepts   = self.world.percepts_at(cx, cy)
            has_breeze = percepts['breeze']
            has_stench = percepts['stench']
            has_glitter= percepts['glitter']
            has_flower = percepts['flower']

            print("\n   Time {} | [{},{}] facing {}   "
                  .format(self.time, cx, cy, orient))
            print("  Percepts: breeze={} stench={} glitter={} flower={}"
                  .format(has_breeze, has_stench, has_glitter, has_flower))

            # Tell percepts to WumpusKB
            self._tell_percepts(cx, cy, has_breeze, has_stench,
                                has_glitter, has_flower)

            # Query KB via pl_fc_entails; log new facts
            self._query_and_log(cx, cy)

            # Grab gold
            if has_glitter and not self.have_gold:
                self._execute(GRAB)
                continue

            # Climb at exit with gold
            if cx == self.dimrow and cy == self.dimrow and self.have_gold:
                self._execute(CLIMB)
                break

            # If holding gold, queue exit path once then let it execute   
            if self.have_gold:                              
                # Try safe visited-only path first, then KB-safe, then risky   
                planned = (                                 
                    self._plan_to_via_visited(self.dimrow, self.dimrow) or   
                    self._plan_to(self.dimrow, self.dimrow, allow_risky=False) or   
                    self._plan_to(self.dimrow, self.dimrow, allow_risky=True)   
                )                                          
                if not planned:                            
                    print("  [No path to exit — halting]") 
                    break                                  
                continue                                   

            # Plan next action; detect and break infinite replanning loops   
            plan_key = (cx, cy, orient, frozenset(self.visited))    
            if plan_key in self._plan_attempts:             
                print("  [Agent stuck — repeated plan at [{},{}]]".format(cx, cy))
                break                                       
            self._plan_attempts.add(plan_key)               
            prev_len = len(self._action_queue)              
            self._plan_next(cx, cy, orient)
            if len(self._action_queue) == prev_len:         
                print("  [Agent stuck — no action possible at [{},{}]]".format(cx, cy))
                break                                       

        if not self.alive:
            pass
        elif not self.exited:
            print("  [Agent halted — no reachable goal]")

        self._print_results()
        return self.actions_log, self.score, self.backtracks


    def _is_kb_safe(self, x, y):                            
        """Ask WumpusKB (pl_fc_entails) if cell (x,y) is safe to enter."""
        return self.kb.is_safe(x, y)                       

    def _is_kb_dangerous(self, x, y):                      
        """Cell is confirmed dangerous if KB proves pit or wumpus there."""
        return self.kb.is_pit(x, y) or self.kb.is_wumpus(x, y)   

    def _safe_unvisited(self):                              
        """Cells adjacent to visited, proven safe by pl_fc_entails, not yet visited."""
        result = []
        for (vx, vy) in self.visited:
            for (nx, ny) in grid_neighbors(vx, vy, self.dimrow):
                if (nx, ny) not in self.visited and self._is_kb_safe(nx, ny):
                    result.append((nx, ny))
        return list(set(result))

    def _risky_frontier(self):                              
        """Unvisited frontier cells that are neither proven safe nor proven dangerous."""
        result = []
        for (vx, vy) in self.visited:
            for (nx, ny) in grid_neighbors(vx, vy, self.dimrow):
                if ((nx, ny) not in self.visited
                        and not self._is_kb_safe(nx, ny)
                        and not self._is_kb_dangerous(nx, ny)):
                    result.append((nx, ny))
        return list(set(result))

    def _passable_set(self, allow_risky=False, extra=None):  
        """Build BFS passable set: visited ∪ KB-safe, minus KB-dangerous."""
        dim = self.dimrow
        passable = set(self.visited)
        for x in range(1, dim + 1):                         
            for y in range(1, dim + 1):                     
                if self._is_kb_safe(x, y):                  
                    passable.add((x, y))                    
        if allow_risky:                                     
            passable |= set(self._risky_frontier())         
        if extra:                                           
            passable.add(extra)                             
        # Strip confirmed dangerous cells
        for x in range(1, dim + 1):                         
            for y in range(1, dim + 1):                     
                if self._is_kb_dangerous(x, y):             
                    passable.discard((x, y))                
        return passable

    def _plan_to_via_visited(self, tx, ty):                

        cx, cy = self.pos.get_location()
        orient = self.pos.get_orientation()
        dim    = self.dimrow

        # Option 1: visited only
        p1 = set(self.visited) | {(tx, ty)}                 
        path = bfs_path((cx, cy), (tx, ty), p1)

        # Option 2: visited + KB-safe
        if path is None:                                    
            path = bfs_path((cx, cy), (tx, ty),
                            self._passable_set(allow_risky=False, extra=(tx, ty)))

        # Option 3: all cells minus confirmed-dangerous (broadest)   
        if path is None:                                    
            all_cells = {(x, y)                             
                         for x in range(1, dim + 1)
                         for y in range(1, dim + 1)}
            p3 = {c for c in all_cells                      
                  if not self._is_kb_dangerous(*c)}         
            p3.add((tx, ty))                                
            path = bfs_path((cx, cy), (tx, ty), p3)         

        if path is None:
            print("  [Planner] No return path to exit found")
            return False
        for a in path_to_actions(path, cx, cy, orient):
            self._action_queue.append(a)
        return True

    def _plan_to(self, tx, ty, allow_risky=False):          
        """BFS-plan path to (tx,ty) and enqueue actions."""
        cx, cy = self.pos.get_location()
        orient = self.pos.get_orientation()
        passable = self._passable_set(allow_risky=allow_risky, extra=(tx, ty))
        path = bfs_path((cx, cy), (tx, ty), passable)
        if path is None:
            return False
        for a in path_to_actions(path, cx, cy, orient):
            self._action_queue.append(a)
        return True

    def _plan_next(self, cx, cy, orient):                   

        dim = self.dimrow

        # 1. Safe unvisited frontier
        safe_front = self._safe_unvisited()
        if safe_front:
            target = min(safe_front,
                         key=lambda c: abs(c[0]-cx) + abs(c[1]-cy))
            self._plan_to(target[0], target[1], allow_risky=False)
            return

        # 2. Shoot at a confirmed wumpus in line of sight
        shot = self._try_shoot_plan(cx, cy, orient)
        if shot:
            for a in shot:
                self._action_queue.append(a)
            return

        # 3. Risky backtrack
        risky = sorted(self._risky_frontier(),
                       key=lambda c: abs(c[0]-cx) + abs(c[1]-cy))
        for target in risky:
            passable = self._passable_set(allow_risky=True, extra=target)
            if bfs_path((cx, cy), target, passable) is not None:
                self.backtracks += 1
                print("  [Backtrack #{}] risking [{},{}]"
                      .format(self.backtracks, target[0], target[1]))
                self._plan_to(target[0], target[1], allow_risky=True)
                return

        # 4. Fall back to exit (or halt if even exit is unreachable)   
        if not self._plan_to(dim, dim, allow_risky=False):     
            # Last resort: try with risky cells allowed          
            if not self._plan_to(dim, dim, allow_risky=True):   
                # Truly stuck — stop gracefully                  
                print("  [Agent stuck — no path anywhere]")     
                self.alive = False                              

    def _try_shoot_plan(self, cx, cy, orient):              
        """Return action list (turns + Shoot) if a KB-confirmed wumpus is in firing line."""
        if not self.have_arrow:                             
            return None
        dim = self.dimrow
        for x in range(1, dim + 1):
            for y in range(1, dim + 1):
                if (x, y) in self.wumpus_dead:
                    continue
                if not self.kb.is_wumpus(x, y):         
                    continue
                # Check line of fire
                if x == cx and y > cy:   needed = NORTH
                elif x == cx and y < cy: needed = SOUTH
                elif y == cy and x > cx: needed = EAST
                elif y == cy and x < cx: needed = WEST
                else: continue
                return steps_to_rotate(orient, needed) + [SHOOT_ACT]   
        return None


    def _execute(self, action):                             
        """Execute one action, update state, tell action to WumpusKB."""
        cx, cy = self.pos.get_location()
        orient = self.pos.get_orientation()
        t      = self.time

        self.actions_log.append(action)
        self.time += 1
        print("  Action[{}]: {}".format(t, action))

        # Score step cost (all actions except Grab/Climb)
        if action not in (GRAB, CLIMB, RELEASE):
            self.score += SCORE_STEP

        #      WumpusKB temporal layer                                                   
        self.kb.add_temporal_sentences(t + 1)              

        #      Tell action sentence to WumpusKB PropKB (temporal)               
        # Maps action string to the corresponding Expr at time t   
        from logic import Expr as _Expr                
        _action_map = {                                     
            FORWARD:    move_forward(t),
            SHOOT_ACT:  shoot(t),
            TURN_LEFT:  turn_left(t),
            TURN_RIGHT: turn_right(t),
            GRAB:       _Expr('Grab', t),
            RELEASE:    _Expr('Release', t),
            CLIMB:      _Expr('Climb', t),
        }
        if action in _action_map:                           
            self.kb.tell(_action_map[action])               
            # Also tell negations of all other actions (make_action_sentence pattern)
            for act_str, act_expr in _action_map.items():   
                if act_str != action:                       
                    self.kb.tell(~act_expr)                 

        #      Physical effects                                                                                 
        if action == TURN_LEFT:
            self.pos.set_orientation(turn_left_dir(orient))

        elif action == TURN_RIGHT:
            self.pos.set_orientation(turn_right_dir(orient))

        elif action == FORWARD:
            nx, ny = forward_cell(cx, cy, orient)
            if 1 <= nx <= self.dimrow and 1 <= ny <= self.dimrow:
                # Safety pre-check: abort if KB now proves dangerous   
                if self._is_kb_dangerous(nx, ny):
                    print("  [Safety abort] KB proves [{},{}] dangerous"
                          .format(nx, ny))
                    self._action_queue.clear()
                    return
                self.pos.set_location(nx, ny)
                new_cell = (nx, ny) not in self.visited
                self.visited.add((nx, ny))
                # Surviving entry => safe                       
                self.kb.fc_tell(no_pit_at(nx, ny))            
                self.kb.fc_tell(no_wumpus_at(nx, ny))         
                self.kb.fc_tell(visited_cell(nx, ny))         
                if new_cell:
                    self._action_queue.clear()                
                # Death checks
                if self.world.has_pit(nx, ny):
                    print("  >>> FELL INTO PIT at [{},{}]! ({})"
                          .format(nx, ny, SCORE_DEATH))
                    self.score += SCORE_DEATH
                    self.alive = False
                elif self.world.has_live_wumpus(nx, ny):
                    print("  >>> EATEN BY WUMPUS at [{},{}]! ({})"
                          .format(nx, ny, SCORE_DEATH))
                    self.score += SCORE_DEATH
                    self.alive = False
            else:
                print("  [Bump] wall")

        elif action == GRAB:                                
            if (cx, cy) == self.world.gold_pos and not self.have_gold:
                self.have_gold = True
                self.score += SCORE_GOLD
                self.kb.fc_tell(have_gold_now())             
                print("  >>> GRABBED GOLD! Score +{}".format(SCORE_GOLD))
                key = "HaveGold"
                if key not in self._logged:
                    self._logged.add(key)
                    self.new_facts_log.append((key, "Agent has the gold"))
                # Plan route to exit — only through VISITED cells (safest route)  
                self._action_queue.clear()
                self._plan_to_via_visited(self.dimrow, self.dimrow)

        elif action == RELEASE:                             
            pass  # drop gold (not needed for goal but action is valid)

        elif action == SHOOT_ACT:                          
            self.score += SCORE_ARROW
            self.have_arrow = False                        
            killed = self._check_kill(cx, cy, orient)
            if killed:
                wx, wy = killed
                self.wumpus_dead.add((wx, wy))              
                self.world.kill_wumpus(wx, wy)
                self.kb.fc_tell(wumpus_dead_at(wx, wy))   
                sym  = "~W{},{}".format(wx, wy)
                desc = "Wumpus killed at [{},{}]".format(wx, wy)
                print("  [KB] {} -- {}".format(sym, desc))
                if sym not in self._logged:
                    self._logged.add(sym)
                    self.new_facts_log.append((sym, desc))
                # Re-query everything after wumpus death
                self._query_and_log(cx, cy)
            else:
                print("  [Shot missed]")

        elif action == CLIMB:                              
            if (cx, cy) == (self.dimrow, self.dimrow):
                if self.have_gold:
                    self.score += SCORE_EXIT
                    self.exited = True
                    print("  >>> EXITED WITH GOLD! Score +{}".format(SCORE_EXIT))
                else:
                    self.exited = True
                    print("  >>> Exited (no gold)")

    def _check_kill(self, cx, cy, orient):                
        """Return (wx,wy) of a wumpus hit by the shot, or None."""
        for (wx, wy) in self.world.wumpus_positions:
            if (wx, wy) in self.wumpus_dead: continue
            if orient == NORTH and wx == cx and wy > cy: return (wx, wy)
            if orient == EAST  and wy == cy and wx > cx: return (wx, wy)
            if orient == SOUTH and wx == cx and wy < cy: return (wx, wy)
            if orient == WEST  and wy == cy and wx < cx: return (wx, wy)
        return None
                                        
    # Output                                                                                              

    def _print_results(self):                              
        print("\n========================================")
        print("SOLUTION PATH (sequence of all actions):")
        for i, a in enumerate(self.actions_log, 1):
            print("  {:3d}. {}".format(i, a))
        print("\nNEW FACTS INFERRED (via pl_fc_entails / WumpusKB):")
        for sym, desc in self.new_facts_log:
            print("  {} -- {}".format(sym, desc))
        print("\nPERFORMANCE:")
        print("  Final Score  : {}".format(self.score))
        print("  Total Steps  : {}".format(self.time))
        print("  Backtracks   : {}".format(self.backtracks))
        print("  Gold Grabbed : {}".format(self.have_gold))
        print("  Exited Safely: {}".format(self.exited))
        print("  Agent Alive  : {}".format(self.alive))
        print("========================================\n")


def main():                                                 
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None  
    print("CSci 384 Wumpus Main (6x6, Random)")
    print("Seed: {}  (for determinstic test output)\n".format(seed))

    world  = WumpusWorldEnv(dimrow=DIMROW, seed=seed)     
    agent  = HybridWumpusAgent(dimrow=DIMROW, world=world) 
    actions, score, backtracks = agent.run()


if __name__ == "__main__":
    main()
