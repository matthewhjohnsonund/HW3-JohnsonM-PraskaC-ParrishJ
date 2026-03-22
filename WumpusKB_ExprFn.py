# 7.2 The Wumpus World

from logic import Expr, disjuncts


def facing_east(time):
    return Expr('FacingEast', time)


def facing_west(time):
    return Expr('FacingWest', time)


def facing_north(time):
    return Expr('FacingNorth', time)


def facing_south(time):
    return Expr('FacingSouth', time)


def wumpus(x, y):
    return Expr('W', x, y)


def pit(x, y):
    return Expr('P', x, y)


def breeze(x, y):
    return Expr('B', x, y)


def stench(x, y):
    return Expr('S', x, y)


def wumpus_alive(time):
    return Expr('WumpusAlive', time)


def have_arrow(time):
    return Expr('HaveArrow', time)


def percept_stench(time):
    return Expr('Stench', time)


def percept_breeze(time):
    return Expr('Breeze', time)


def percept_glitter(time):
    return Expr('Glitter', time)


def percept_bump(time):
    return Expr('Bump', time)


def percept_scream(time):
    return Expr('Scream', time)


def move_forward(time):
    return Expr('Forward', time)


def shoot(time):
    return Expr('Shoot', time)


def turn_left(time):
    return Expr('TurnLeft', time)


def turn_right(time):
    return Expr('TurnRight', time)


def ok_to_move(x, y, time):
    return Expr('OK', x, y, time)


def location(x, y, time=None):
    if time is None:
        return Expr('L', x, y)
    else:
        return Expr('L', x, y, time)


# #Extended – new sensor for exit/flower adjacency
def flower(x, y):
    return Expr('Flower', x, y)


def exit_sq(x, y):
    return Expr('Exit', x, y)


def gold(x, y):
    return Expr('Gold', x, y)


def have_gold(time):
    return Expr('HaveGold', time)


def percept_flower(time):
    return Expr('PerceptFlower', time)

def no_breeze_at(x, y):   return Expr('NoBreezeAt', x, y)    
def breeze_at(x, y):      return Expr('BreezeAt', x, y)      
def no_stench_at(x, y):   return Expr('NoStenchAt', x, y)    
def stench_at(x, y):      return Expr('StenchAt', x, y)      
def visited_cell(x, y):   return Expr('Visited', x, y)       
def no_pit_at(x, y):      return Expr('NoPit', x, y)         
def pit_at(x, y):         return Expr('PitAt', x, y)         
def no_wumpus_at(x, y):   return Expr('NoWumpus', x, y)      
def wumpus_at(x, y):      return Expr('WumpusAt', x, y)      
def safe_cell(x, y):      return Expr('Safe', x, y)          
def wumpus_dead_at(x, y): return Expr('WumpusDead', x, y)    
def have_gold_now():       return Expr('HaveGoldNow')         


# Symbols

def implies(lhs, rhs):
    return Expr('==>', lhs, rhs)


def equiv(lhs, rhs):
    return Expr('<=>', lhs, rhs)


# Helper Function

def new_disjunction(sentences):
    t = sentences[0]
    for i in range(1, len(sentences)):
        t |= sentences[i]
    return t
