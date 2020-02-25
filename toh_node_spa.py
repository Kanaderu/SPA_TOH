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

def toh_node_create(disk_count, D, vocab):
    toh = TowerOfHanoi(disk_count, D, vocab)
    with nengo.Network(label="Tower of Hanoi node") as toh_n:
        
        toh_n.toh_node = nengo.Node(toh, size_in=5*D, size_out=7*D + 6, label='TOH Node')
        
        # connect inputs
        toh_n.focus_in = nengo.Node(size_in=D)
        toh_n.goal_peg_in = nengo.Node(size_in=D)
        toh_n.goal_in = nengo.Node(size_in=D)
        toh_n.move_peg_in = nengo.Node(size_in=D)
        toh_n.move_in = nengo.Node(size_in=D)
        nengo.Connection(toh_n.focus_in, toh_n.toh_node[:D])
        nengo.Connection(toh_n.goal_peg_in, toh_n.toh_node[D:2*D])
        nengo.Connection(toh_n.goal_in, toh_n.toh_node[2*D:3*D])
        nengo.Connection(toh_n.move_peg_in, toh_n.toh_node[3*D:4*D])
        nengo.Connection(toh_n.move_in, toh_n.toh_node[4*D:5*D])
        
        # connect outputs
        toh_n.largest_out = nengo.Node(size_in=D)
        toh_n.focus_out = nengo.Node(size_in=D)
        toh_n.goal_peg_out = nengo.Node(size_in=D)
        toh_n.target_peg_out = nengo.Node(size_in=D)
        toh_n.goal_out = nengo.Node(size_in=D)
        toh_n.goal_peg_final_out = nengo.Node(size_in=D)
        toh_n.focus_peg_out = nengo.Node(size_in=D)
        nengo.Connection(toh_n.toh_node[:D], toh_n.largest_out)
        nengo.Connection(toh_n.toh_node[D:2*D], toh_n.focus_out)
        nengo.Connection(toh_n.toh_node[2*D:3*D], toh_n.goal_peg_out)
        nengo.Connection(toh_n.toh_node[3*D:4*D], toh_n.target_peg_out)
        nengo.Connection(toh_n.toh_node[4*D:5*D], toh_n.goal_out)
        nengo.Connection(toh_n.toh_node[5*D:6*D], toh_n.goal_peg_final_out)
        nengo.Connection(toh_n.toh_node[6*D:7*D], toh_n.focus_peg_out)
        
        # connect visualization
        toh_n.focus_viz = nengo.Node(size_in=1)#, label="focus disk")
        toh_n.goal_viz = nengo.Node(size_in=1)#, label="goal disk")
        toh_n.peg_viz = nengo.Node(size_in=1)#, label="goal peg")
        toh_n.pos_viz = nengo.Node(size_in=3)
        nengo.Connection(toh_n.toh_node[-6], toh_n.focus_viz)
        nengo.Connection(toh_n.toh_node[-5], toh_n.goal_viz)
        nengo.Connection(toh_n.toh_node[-4], toh_n.peg_viz)
        nengo.Connection(toh_n.toh_node[-3:], toh_n.pos_viz)
        
        
    return toh_n


class TowerOfHanoi(object):
    def __init__(self, disk_count, D, vocab):
        self.D = D
        self.pstc = 0.01
        self.disk_count = disk_count
        
        self.pegs = [vocab.parse('A'), vocab.parse('B'), vocab.parse('C')]
        self.disks = [vocab.parse('D{}'.format(i)) for i in range(self.disk_count)] + [vocab.parse('NONE')]
        self.reset()
        self.location_dict = {'A':0, 'B':1, 'C':2}  # location_dict[toh.target_peg] -> goal peg
        self.zero = [0]*D
        self.vocab = vocab
        
    def reset(self, randomize=True):
        self.location = ['A']*self.disk_count
        #self.location=['A','C','C']
        self.focus = len(self.disks) - 1    # focus disk
        self.largest = self.disk_count - 1
        self.goal = 2                       # goal disk
        self.target_peg = 'C'
        self.move_peg_data = [0] * self.disk_count
        self.goal_peg_data = [0] * self.disk_count
        # constant
        self.target = ['C'] * self.disk_count
        
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
        
    def __call__(self, t, x):
        #######
        # Input
        #######
        focus_in = x[:self.D]
        goal_peg = x[self.D:2*self.D]
        goal_in = x[2*self.D:3*self.D]
        
        # motor cortex input
        move_peg = x[3*self.D:4*self.D]
        move = x[4*self.D:5*self.D]
        
        ############
        # Processing
        ############
        self.focus = np.argmax(spa.similarity(focus_in, self.disks))
        self.goal_peg_data = spa.similarity(goal_peg, self.disks)
        
        ##
        disks = spa.similarity(goal_in, self.disks)
        pegs = self.goal_peg_data
        if np.max(pegs) > threshold and np.max(disks) > threshold:
            self.goal = np.argmax(disks)
            self.target_peg = 'ABC'[np.argmax(pegs)]
        
        self.move_peg_data = spa.similarity(move_peg, self.pegs)
        ##
        
        ##
        disks = spa.similarity(move, self.disks)
        disk = np.argmax(disks)
        pegs = self.move_peg_data
        peg = 'ABC'[np.argmax(pegs)] # 'ABC' is a char array
        
        if(np.max(pegs) > threshold and np.max(disks) > threshold):
            if peg != self.peg(disk):
                if self.can_move(disk, peg):
                    self.move(disk, peg)
                    print('Moving D{} to {}'.format(disk, peg))
                else:    
                    print('Cannot move D{} to {}'.format(disk,peg))
        ##
        ########
        # Output
        ########
        
        # define output array
        out = [0]*7*self.D
        out[:self.D] = self.disks[self.largest].v # largest
        out[self.D:2*self.D] = self.disks[self.focus].v # focus_out
        out[2*self.D:3*self.D] = self.vocab.parse(self.peg(self.goal)).v # goal_peg_out
        out[3*self.D:4*self.D] = self.vocab.parse(self.target_peg).v # target_peg
        
        # visual cortex output
        out[4*self.D:5*self.D] = self.disks[self.goal].v # goal_out
        out[5*self.D:6*self.D] = self.vocab.parse(self.target[self.goal]).v # goal_peg_final
        out[6*self.D:7*self.D] = self.zero if self.focus >= self.disk_count else self.vocab.parse(self.peg(self.focus)).v # focus_peg
        
        out_viz = [0]*(3 + len(self.location))
        out_viz[0] = self.focus                             # focus_viz
        out_viz[1] = self.goal                              # goal_viz
        out_viz[2] = self.location_dict[self.target_peg]    # peg_viz
        for idx, loc in enumerate(self.location):
            out_viz[3 + idx] = self.location_dict[loc]      # pos_viz
        out += out_viz
        return out
    
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
        
    viz_node = nengo.Node(viz_func, size_in=6)
    nengo.Connection(hanoi_node.focus_viz, viz_node[0])
    nengo.Connection(hanoi_node.goal_viz, viz_node[1])
    nengo.Connection(hanoi_node.peg_viz, viz_node[2])
    nengo.Connection(hanoi_node.pos_viz, viz_node[3:6])
    
    # connect all state inputs for visual
    move_in_state = spa.State(vocab, dimensions)
    move_peg_in_state = spa.State(vocab, dimensions)
    goal_in_state = spa.State(vocab, dimensions)
    goal_peg_in_state = spa.State(vocab, dimensions)
    focus_in_state = spa.State(vocab, dimensions)
    nengo.Connection(move_in_state.output, hanoi_node.move_in, synapse=None)
    nengo.Connection(move_peg_in_state.output, hanoi_node.move_peg_in, synapse=None)
    nengo.Connection(goal_in_state.output, hanoi_node.goal_in, synapse=None)
    nengo.Connection(goal_peg_in_state.output, hanoi_node.goal_peg_in, synapse=None)
    nengo.Connection(focus_in_state.output, hanoi_node.focus_in, synapse=None)
    
    # connect all state outputs for visual
    largest_out_state = spa.State(vocab, dimensions)
    focus_out_state = spa.State(vocab, dimensions)
    goal_peg_out_state = spa.State(vocab, dimensions)
    target_peg_out_state = spa.State(vocab, dimensions)
    goal_out_state = spa.State(vocab, dimensions)
    goal_peg_final_out_state = spa.State(vocab, dimensions)
    focus_peg_out_state = spa.State(vocab, dimensions)
    nengo.Connection(hanoi_node.largest_out, largest_out_state.input, synapse=None)
    nengo.Connection(hanoi_node.focus_out, focus_out_state.input, synapse=None)
    nengo.Connection(hanoi_node.goal_peg_out, goal_peg_out_state.input, synapse=None)
    nengo.Connection(hanoi_node.target_peg_out, target_peg_out_state.input, synapse=None)
    nengo.Connection(hanoi_node.goal_out, goal_out_state.input, synapse=None)
    nengo.Connection(hanoi_node.goal_peg_final_out, goal_peg_final_out_state.input, synapse=None)
    nengo.Connection(hanoi_node.focus_peg_out, focus_peg_out_state.input, synapse=None)