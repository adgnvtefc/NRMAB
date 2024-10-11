import random

class SimpleNode:
    def __init__(self, passive_activation_chance: float, passive_deactivation_chance: float, active_activation_chance: float, active_deactivation_chance: float, can_cascade = True):
        if 0 <= passive_activation_chance <= 1:
            self.passive_activation_chance = passive_activation_chance
        else:
            raise ValueError("activation_chance must be between 0 and 1")
        
        if 0 <= passive_deactivation_chance <= 1:
            self.passive_deactivation_chance = passive_deactivation_chance
        else:
            raise ValueError("deactivation_chance must be between 0 and 1")
        
        if 0 <= passive_activation_chance <= 1:
            self.active_activation_chance = active_activation_chance
        else:
            raise ValueError("activation_chance must be between 0 and 1")
        
        if 0 <= passive_deactivation_chance <= 1:
            self.active_deactivation_chance = active_deactivation_chance
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
    
    def getPassiveActivationChance(self):
        return self.passive_activation_chance
    
    def getPassiveDeactivationChance(self):
        return self.passive_deactivation_chance
    
    def getActiveActivationChance(self):
        return self.active_activation_chance
    
    def getActiveDeactivationChance(self):
        return self.active_deactivation_chance
    
    def rearm(self):
        self.can_cascade = True

    def __hash__(self):
        """
        Generate a unique hash based on a combination of:
        1. A unique ID assigned to each node.
        """
        return hash((self._unique_id))
    
    def __repr__(self):
        return f"Node(active={self.active}, passive_activation_chance={self.passive_activation_chance}, active_activation_chance={self.active_activation_chance})"

    def __eq__(self, other):
        """Check equality between two Node objects."""
        if isinstance(other, SimpleNode):
            return (self._unique_id, self.active, self.active_activation_chance) == (other._unique_id, other.active, other.active_activation_chance)
        return False