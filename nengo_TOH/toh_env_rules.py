import nengo
import nengo_spa as spa
from toh_node_spa import *

vocab_dim = 32

model = nengo.Network('TOH')

def toh_agent():
    model = nengo.Network('ToH Agent')
    with model:
        env = nengo.Node()
        # Table E.1 is used to define the spa states and subnetworks 
        # of the cortical elements for the Tower of Hanoi model
        with nengo.Network('buffer') as model.buffer:
            # used to control the different stages of the problem-solving algorithm
            model.buffer.state = spa.State(vocab=vocab_dim)
            # stores the disk currently being attended to (D0, D1, D2, D3)
            model.buffer.focus = spa.State(vocab=vocab_dim)
            # stores the disk we are trying to move (D0, D1, D2, D3)
            model.buffer.goal = spa.State(vocab=vocab_dim)
            # stores the location we want to move the goal disk to (A, B, C)
            model.buffer.goal_target = spa.State(vocab=vocab_dim)
        
        with nengo.Network('sensory') as model.sensory:
            # automatically contains the location of the focus disk (A, B, C)
            model.sensory.focus_peg = spa.State(vocab=vocab_dim)
            # automatically contains the location of the goal disk (A, B, C)
            model.sensory.goal_current = spa.State(vocab=vocab_dim)
            # automatically contains the final desired location of the goal disk (A, B, C)
            model.sensory.goal_final = spa.State(vocab=vocab_dim)
            # automatically contains the largest visible disk (D3)
            model.sensory.largest = spa.State(vocab=vocab_dim)
            # automcatically contains DONE if the motor action is finished
            model.sensory.motor = spa.State(vocab=vocab_dim)
            
        with nengo.Network('memory') as model.memory:
            # stores an association between mem1 and mem2 in working memory
            model.memory.mem_1 = spa.State(vocab=vocab_dim)
            # stores an association between mem1 and mem2 in working memory
            model.memory.mem_2 = spa.State(vocab=vocab_dim)
            # indicates one element of a pair to attempt to recall from working memory
            model.memory.request = spa.State(vocab=vocab_dim)
            # the vector associated with the currently requested vector
            model.memory.recall = spa.State(vocab=vocab_dim)
            
        with nengo.Network('motor') as model.motor:
            # tells the motor system which disk to move (A, B, C)
            model.motor.move_disk = spa.State(vocab=vocab_dim)
            # tells the motor system where to move the disk to (A, B, C)
            model.motor.move_peg = spa.State(vocab=vocab_dim)
            
        # Table E.2 is used to define the spa rules for the Tower of Hanoi model
        with nengo.Network('TOH Rules') as model.rules:
            with spa.ActionSelection() as model.rules.action_sel:
                spa.ifmax('LookDone',
                    -spa.dot(model.buffer.focus, spa.sym.D0) +
                    spa.dot(model.buffer.goal, model.buffer.focus) + 
                    spa.dot(model.sensory.goal_current, model.buffer.goal_target) + 
                    spa.dot(model.buffer.state, spa.sym.STORE),
                    model.buffer.goal * spa.sym.NEXT >> model.buffer.focus,
                    model.buffer.goal * spa.sym.NEXT >> model.buffer.goal,
                    model.sensory.goal_final >> model.buffer.goal_target)
                spa.ifmax('LookNotDone',
                    -spa.dot(model.buffer.focus, spa.sym.D0) +
                    spa.dot(model.buffer.goal, model.buffer.focus) + 
                    -spa.dot(model.sensory.goal_current, model.buffer.goal_target) + 
                    spa.dot(model.buffer.state, spa.sym.STORE),
                    model.buffer.goal * spa.sym.NEXT >> model.buffer.focus)
                spa.ifmax('InTheWay1', 
                    -spa.dot(model.buffer.focus, model.buffer.goal) + 
                    spa.dot(model.sensory.focus_peg, model.sensory.goal_current) + 
                    -spa.dot(model.sensory.focus_peg, model.buffer.goal_target) + 
                    -spa.dot(model.buffer.state, spa.sym.STORE),
                    model.buffer.goal * spa.sym.NEXT >> model.buffer.focus)
                spa.ifmax('InTheWay2',
                    -spa.dot(model.buffer.focus, model.buffer.goal) + 
                    -spa.dot(model.sensory.focus_peg, model.sensory.goal_current) + 
                    spa.dot(model.sensory.focus_peg, model.buffer.goal_target) + 
                    -spa.dot(model.buffer.state, spa.sym.STORE),
                    model.buffer.goal * spa.sym.NEXT >> model.buffer.focus)
                spa.ifmax('NotInTheWay',
                    -spa.dot(model.buffer.focus, model.buffer.goal) + 
                    -spa.dot(model.sensory.focus_peg, model.sensory.goal_current) + 
                    -spa.dot(model.sensory.focus_peg, model.buffer.goal_target) + 
                    -spa.dot(model.buffer.focus, spa.sym.D0),
                    model.buffer.goal * spa.sym.NEXT >> model.buffer.focus)
                
                spa.ifmax('MoveD0',
                    spa.dot(model.buffer.focus, spa.sym.D0) + 
                    spa.dot(model.buffer.goal, spa.sym.D0) + 
                    -spa.dot(model.sensory.goal_current, model.buffer.goal_target),
                    spa.sym.D0 >> model.motor.move_disk,
                    model.buffer.goal_target >> model.motor.move_peg)
                spa.ifmax('MoveGoal',
                    spa.dot(model.buffer.focus, spa.sym.D0) + 
                    -spa.dot(model.buffer.goal, spa.sym.D0) + 
                    -spa.dot(model.sensory.focus_peg, model.buffer.goal_target) + 
                    -spa.dot(model.buffer.goal_target, model.sensory.goal_current) + 
                    -spa.dot(model.sensory.focus_peg, model.sensory.goal_current),
                    model.buffer.goal >> model.motor.move_disk,
                    model.buffer.goal_target >> model.motor.move_peg)
                spa.ifmax('MoveDone',
                    spa.dot(model.sensory.motor, spa.sym.DONE) + 
                    -spa.dot(model.buffer.goal, model.sensory.largest) + 
                    -spa.dot(model.buffer.state, spa.sym.RECALL),
                    spa.sym.RECALL >> model.buffer.state,
                    model.buffer.goal * ~spa.sym.NEXT >> model.buffer.goal)
                spa.ifmax('MoveDone2',
                    spa.dot(model.sensory.motor, spa.sym.DONE) + 
                    spa.dot(model.buffer.goal, model.sensory.largest) + 
                    -spa.dot(model.buffer.state, spa.sym.RECALL),
                    model.sensory.largest * ~spa.sym.NEXT >> model.buffer.focus,
                    model.sensory.largest * ~spa.sym.NEXT >> model.buffer.goal,
                    model.sensory.goal_final >> model.buffer.goal_target,
                    spa.sym.HANOI >> model.buffer.state)
                    
                spa.ifmax('Store',
                    spa.dot(model.buffer.state, spa.sym.STORE) + 
                    -spa.dot(model.memory.recall, model.buffer.goal_target),
                    model.buffer.goal >> model.memory.mem_1,
                    model.buffer.goal_target >> model.memory.mem_2,
                    model.buffer.goal >> model.memory.request)
                spa.ifmax('StoreDone',
                    spa.dot(model.buffer.state, spa.sym.STORE) + 
                    spa.dot(model.memory.recall, model.buffer.goal_target),
                    spa.sym.FIND >> model.buffer.state)
                
                spa.ifmax('FindFree1',
                    spa.dot(model.buffer.state, spa.sym.FIND) + 
                    -spa.dot(model.buffer.focus, model.buffer.goal) + 
                    spa.dot(model.sensory.focus_peg, model.sensory.goal_current) + 
                    -spa.dot(model.sensory.focus_peg, model.buffer.goal_target),
                    spa.sym.A + spa.sym.B + spa.sym.C - model.sensory.focus_peg - model.buffer.goal_target >> model.buffer.goal_target,
                    model.buffer.focus >> model.buffer.goal,
                    spa.sym.HANOI >> model.buffer.state)
                spa.ifmax('FindFree2',
                    spa.dot(model.buffer.state, spa.sym.FIND) + 
                    -spa.dot(model.buffer.focus, model.buffer.goal) + 
                    -spa.dot(model.sensory.focus_peg, model.sensory.goal_current) + 
                    spa.dot(model.sensory.focus_peg, model.buffer.goal_target),
                    spa.sym.A + spa.sym.B + spa.sym.C - model.sensory.goal_current - model.buffer.goal_target >> model.buffer.goal_target,
                    model.buffer.focus >> model.buffer.goal,
                    spa.sym.HANOI >> model.buffer.state)
                
                spa.ifmax('Recall',
                    spa.dot(model.buffer.state, spa.sym.RECALL) + 
                    -spa.dot(model.memory.recall, spa.sym.A + spa.sym.B + spa.sym.C),
                    model.buffer.goal >> model.memory.request)
                spa.ifmax('RecallDo',
                    spa.dot(model.buffer.state, spa.sym.RECALL) + 
                    spa.dot(model.memory.recall, spa.sym.A + spa.sym.B + spa.sym.C) +
                    -spa.dot(model.memory.recall, model.sensory.goal_current),
                    spa.sym.HANOI >> model.buffer.state,
                    model.buffer.goal >> model.buffer.focus,
                    4 * model.memory.recall >> model.buffer.goal_target)
                spa.ifmax('RecallNext',
                    spa.dot(model.buffer.state, spa.sym.RECALL) + 
                    spa.dot(model.memory.recall, spa.sym.A + spa.sym.B + spa.sym.C) +
                    spa.dot(model.memory.recall, model.sensory.goal_current),
                    spa.sym.HANOI >> model.buffer.state,
                    model.buffer.goal * ~spa.sym.NEXT >> model.buffer.goal,
                    model.buffer.goal >> model.memory.request)
    return model
        
with model:
    agent = toh_agent()
    vis_network = toh_vis_debug('Tower of Hanoi')