#from .fixed_nmp import PhysicsNMP
#from .fixed_nmp import PhysicsStackNMP
#from .fixed_nmp import OnesNMP
#from .fixed_nmp import EyeNMP
#from .fixed_nmp import LearnedFixedNMP
#from .fixed_nmp import PhysicsPlusLearnedNMP
from .fixed_nmp import FixedNMP

'''
This module contains NMPs that use a single, fixed adjacency matrix to pass
messages. The matrix is kept constant across layers/iterations of message passing.
This matrix can be learned from the inputs or given.
'''
