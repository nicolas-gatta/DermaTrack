class RunningAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0
        self.values = []

    def update(self, value):
        """
        Updates the running average with a new value.
        """
        self.total += value
        self.count += 1
        self.values.append(value)
        
    def reset(self):
        """
        Reset all the value of the running average.
        """
        self.total = 0.0
        self.count = 0.0

    @property
    def average(self) -> float:
        """
        Returns the current average.
        """
        if self.count == 0.0:
            return 0.0 
        return self.total / self.count
    
    @property
    def all_values(self) -> list:
        return self.values
