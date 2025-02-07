import random

class SimpleNode:
    def __init__(self, active_activation_active_action: float, active_activation_passive_action: float, passive_activation_active_action: float, passive_activation_passive_action: float, value: int, can_cascade = True):

        #BIG CHANCE
        self.active_activation_active = active_activation_active_action

        #SMALLER CHANCE
        self.active_activation_passive = active_activation_passive_action

        #BIG CHANCE
        self.passive_activation_active = passive_activation_active_action

        #SMALLER CHANCE
        self.passive_activation_passive = passive_activation_passive_action
        
        self.value = value
        
        self.active = False
        self._unique_id = random.randint(0, 10**6)  # Adding a unique identifier
        self.can_cascade = can_cascade


    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False
    
    def getValue(self):
        return self.value
    
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
    
    def transition(self, action:bool):
        if self.active:
            if action:
                if random.random() < self.active_activation_active:
                    self.active = True
                else:
                    self.active = False
            else:
                if random.random() < self.active_activation_passive:
                    self.active = True
                else:
                    self.active = False
        else:
            if action:
                if random.random() < self.passive_activation_active:
                    self.active = True
                else:
                    self.active = False
            else:
                if random.random() < self.passive_activation_passive:
                    self.active = True
                else:
                    self.active = False
        return self.isActive()

    
    def rearm(self):
        self.can_cascade = True

    def __hash__(self):
        """
        Generate a unique hash based on a combination of:
        1. A unique ID assigned to each node.
        """
        return hash((self._unique_id))
    
    def __repr__(self):
        return (
            f"SimpleNode(active={self.active}, "
            f"active_activation_active={self.active_activation_active}, "
            f"active_activation_passive={self.active_activation_passive}, "
            f"passive_activation_active={self.passive_activation_active}, "
            f"passive_activation_passive={self.passive_activation_passive}, "
            f"value={self.value})"
        )
    def __eq__(self, other):
        """Check equality between two SimpleNode objects."""
        if isinstance(other, SimpleNode):
            return (
                self._unique_id == other._unique_id and
                self.active == other.active and
                self.active_activation_active == other.active_activation_active and
                self.active_activation_passive == other.active_activation_passive and
                self.passive_activation_active == other.passive_activation_active and
                self.passive_activation_passive == other.passive_activation_passive
            )
        return False