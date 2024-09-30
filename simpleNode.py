import random

class SimpleNode:
    def __init__(self, activation_chance: float, deactivation_chance: float, can_cascade = True):
        if 0 <= activation_chance <= 1:
            self.activation_chance = activation_chance
        else:
            raise ValueError("activation_chance must be between 0 and 1")
        
        if 0 <= deactivation_chance <= 1:
            self.deactivation_chance = deactivation_chance
        else:
            raise ValueError("deactivation_chance must be between 0 and 1")
        
        self.active = False
        self._unique_id = random.randint(0, 10**6)  # Adding a unique identifier
        self.can_cascade = can_cascade


    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False
    
    def isActive(self):
        return self.active == True
    
    def canCascade(self):
        return self.can_cascade
    
    def cascade(self):
        if self.can_cascade == True:
            self.can_cascade = False
            return True
        else:
            return False

    def __hash__(self):
        """
        Generate a unique hash based on a combination of:
        1. A unique ID assigned to each node.
        """
        return hash((self._unique_id))
    
    def __repr__(self):
        return f"Node(active={self.active}, activation_chance={self.activation_chance})"

    def __eq__(self, other):
        """Check equality between two Node objects."""
        if isinstance(other, SimpleNode):
            return (self._unique_id, self.active, self.activation_chance) == (other._unique_id, other.active, other.activation_chance)
        return False