class RunningAverage:
    def __init__(self):
        self.total = 0.0  # Sum of all values added
        self.count = 0    # Number of values added

    def update(self, value):
        """
        Updates the running average with a new value.
        """
        self.total += value
        self.count += 1
        
    def reset(self):
        """
        Reset all the value of the running average.
        """
        self.total = 0
        self.count = 0.0

    @property
    def average(self):
        """
        Returns the current average.
        """
        if self.count == 0:
            return 0.0  # Avoid division by zero
        return self.total / self.count
