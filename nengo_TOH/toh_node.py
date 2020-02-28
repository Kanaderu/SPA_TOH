import numpy as np
import nengo

import ipdb

import nengo_spa as spa

np.random.seed(0)

# constants
dimensions = 9
disk_count = 3
threshold = 0.5

vocab = spa.Vocabulary(dimensions)
vocab.populate('A; B; C; D0; D1; D2; D3; NONE')

def get_similarity_array(x, thingys):
    return [np.dot(x, thing.v) for thing in thingys]

def toh_node_create(disk_count, D, vocab):
    with nengo.Network(label="Tower of Hanoi node") as toh_n:
        toh = TowerOfHanoi(disk_count, D, vocab)
        
        #### input nodes ####
        def focus_in_func(t, x):
            toh.focus = np.argmax(spa.similarity(x, toh.disks))
            print(toh)
        toh_n.focus_in = nengo.Node(focus_in_func, size_in=D)
        
        def goal_peg_func(t, x):
            toh.goal_peg_data = spa.similarity(x, toh.disks)
        toh_n.goal_peg = nengo.Node(goal_peg_func, size_in=D)
        
        def goal_in_func(t, x):
            """Sets the goals for both the pegs and the disk"""
            disks = spa.similarity(x, toh.disks)
            pegs = toh.goal_peg_data
            print('target_peg = {}'.format(toh.target_peg))
            if np.max(pegs) > threshold and np.max(disks) > threshold:
                toh.goal = disks.index(np.max(disks))
                toh.target_peg = 'ABC'[pegs.index(np.max(pegs))]
        toh_n.goal_in = nengo.Node(goal_in_func, size_in=D)
        
        toh_n.motor = nengo.Network(label="Motor Cortex")
        with toh_n.motor as motor:
            def move_peg_func(t, x):
                toh.move_peg_data = spa.similarity(x, toh.pegs)
            motor.move_peg = nengo.Node(move_peg_func, size_in=D)
            
            def move_func(t, x):
                disks = spa.similarity(x, toh.disks)
                disk = np.argmax(disks)
                pegs = toh.move_peg_data
                peg = 'ABC'[np.argmax(pegs)] # 'ABC' is a char array
                
                if(np.max(pegs) > threshold and np.max(disks) > threshold):
                    if peg != toh.peg(disk):
                        if toh.can_move(disk, peg):
                            toh.move(disk, peg)
                            print('Moving D{} to {}'.format(disk, peg))
                        else:    
                            print('Cannot move D{} to {}'.format(disk,peg))
                            
            motor.move = nengo.Node(move_func, size_in=D)
            
        #### output nodes ####
        toh_n.largest = nengo.Node(lambda t: toh.disks[toh.largest].v)
        toh_n.focus_out = nengo.Node(lambda t: toh.disks[toh.focus].v)
        # HOW THE HELL IS THE GOAL EVEN BEING SET TO NONE
        toh_n.goal_peg_out = nengo.Node(lambda t: vocab.parse(toh.peg(toh.goal)).v)
        
        toh_n.target_peg = nengo.Node(lambda t: vocab.parse(toh.target_peg).v)
        
        # This just says where all the disks are supposed to end up. In this case, at the last peg
        # Put this into the visual cortex, then check the rules are flowing correctly
        toh_n.vis = nengo.Network(label="Visual Cortex")
        
        with toh_n.vis as vis:
            vis.goal_out = nengo.Node(lambda t: toh.disks[toh.goal].v)
            vis.goal_peg_final = nengo.Node(lambda t: vocab.parse(toh.target[toh.goal]).v)
            
            def focus_peg_func(t):
                if toh.focus >= disk_count:
                    return toh.zero
                return vocab.parse(toh.peg(toh.focus)).v
                
            vis.focus_peg = nengo.Node(focus_peg_func(toh))
        ####
        toh_n.largest_out_state = spa.State(vocab, D)
        toh_n.focus_out_state = spa.State(vocab, D)
        toh_n.goal_peg_out_state = spa.State(vocab, D)
        toh_n.target_peg_out_state = spa.State(vocab, D)
        
        nengo.Connection(toh_n.largest, toh_n.largest_out_state.input, synapse=None)
        nengo.Connection(toh_n.focus_out, toh_n.focus_out_state.input, synapse=None)
        nengo.Connection(toh_n.goal_peg_out, toh_n.goal_peg_out_state.input, synapse=None)
        nengo.Connection(toh_n.target_peg, toh_n.target_peg_out_state.input, synapse=None)
        
        #### Visualization nodes ####

        #toh_n.focus_viz = nengo.Node(lambda t: toh.focus, size_out=1, label="focus disk")
        #toh_n.goal_viz = nengo.Node(lambda t: toh.goal, size_out=1, label="goal disk")
        #toh_n.peg_viz = nengo.Node(lambda t: toh.location_dict[toh.target_peg], size_out=1, label="goal peg")

        def pos_viz_func(t):
            return_val = [0]*3
            for r_i, loc in enumerate(toh.location):
                return_val[r_i] = toh.location_dict[loc]
            return return_val
            
        toh_n.pos_viz = nengo.Node(pos_viz_func, size_out=3)
        
    return toh_n


class TowerOfHanoi(object):
    def __init__(self, disk_count, D, vocab):
        self.pstc = 0.01
        
        self.pegs = [vocab.parse('A'), vocab.parse('B'), vocab.parse('C')]
        self.disks = [vocab.parse('D{}'.format(i)) for i in range(disk_count)] + [vocab.parse('NONE')]
        self.reset()
        self.location_dict = {'A':0, 'B':1, 'C':2}
        self.zero = [0]*D
        self.vocab = vocab
        
    def reset(self, randomize=True):
        self.location = ['A']*disk_count
        #self.location=['A','C','C']
        self.focus = len(self.disks) - 1
        self.largest = disk_count - 1
        self.goal = 2
        self.target_peg = 'C'
        self.move_peg_data = [0] * disk_count
        self.goal_peg_data = [0] * disk_count
        # constant
        self.target = ['C'] * disk_count
        
    def move(self, disk, peg):
        assert self.can_move(disk,peg)
        self.location[disk] = peg
        
    def peg(self, disk):
        try:
            return self.location[disk]
        except IndexError:
            ipdb.set_trace()
        
    def can_move(self, disk, peg):
        assert peg in 'ABC'
        pegs = [self.peg(disk), peg]
        for i in range(disk):
            if self.peg(i) in pegs:
                return False
        return True
    
    def __str__(self):
        res = '=============================\nTOH Stats\n============================='
        res += '\n    location: {}'.format(self.location)
        res += '\n    focus: {}'.format(self.focus)
        res += '\n    largest: {}'.format(self.largest)
        res += '\n    goal: {}'.format(self.goal)
        res += '\n    target_peg: {}'.format(self.target_peg)
        res += '\n    move_peg_data: {}'.format(self.move_peg_data)
        res += '\n    goal_peg_data: {}'.format(self.goal_peg_data)
        res += '\n    target: {}'.format(self.target)
        res += '\n============================='
        return res

# focus peg is blue
# goal disc is blue
# focus disk is red
# when goal == focus, the disk is purple

model = nengo.Network('TOH Node')
with model:
    hanoi_node = toh_node_create(disk_count, dimensions, vocab)
    
    ##### Node for visualization #####
    def viz_func(t, x):
        focus_peg = [0]*3
        if x[0] < 3:
            focus_peg[int(x[0])] = 255
        print("focus_peg: {}".format(focus_peg))
        
        goal_disc = [0]*3
        if x[1] < 3:
            goal_disc[int(x[1])] = 255
        print("goal_disc: {}".format(goal_disc))
        
        focus_disc = [0]*3
        if x[1] < 3:
            focus_disc[int(x[2])] = 255
        print("focus_disc: {}".format(focus_disc))
        
        location = x[3:6]
        viz_func._nengo_html_ = '''
        <svg width="400" height="110">
          <rect x="50" y="0" width="10" height="600" style="fill:rgb(0,0,%i);" />
          <rect x="150" y="0" width="10" height="600" style="fill:rgb(0,0,%i);" />
          <rect x="250" y="0" width="10" height="600" style="fill:rgb(0,0,%i);" />
          
          <rect x="%i" y="40" width="40" height="20" style="fill:rgb(%i,0,%i);" />
          <rect x="%i" y="70" width="70" height="15" style="fill:rgb(%i,0,%i);" />
          <rect x="%i" y="100" width="100" height="10" style="fill:rgb(%i,0,%i);" />
        </svg>
        ''' %(focus_peg[0], focus_peg[1], focus_peg[2],
             (35+location[0]*100), focus_disc[0], goal_disc[0],
             (15+location[1]*100), focus_disc[1], goal_disc[1],
             (location[2]*100), focus_disc[2], goal_disc[2])
        
    viz_node = nengo.Node(viz_func, size_in=7)
    '''
    nengo.Connection(hanoi_node.focus_viz, viz_node[0])
    nengo.Connection(hanoi_node.goal_viz, viz_node[1])
    nengo.Connection(hanoi_node.peg_viz, viz_node[2])
    nengo.Connection(hanoi_node.pos_viz, viz_node[3:6])
    '''