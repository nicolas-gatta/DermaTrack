class EarlyStopping:
    
    """
    A class to implement early stopping during training to prevent overfitting.
    """
    
    def __init__(self, patience = 10, delta = 0.0, verbose = True):
        """
        Initialize the EarlyStopping Class

        args:
            __patience (int): Number of epochs to wait for improvement before stopping. Default = 10.
            __delta (float): Minimum change in the monitored loss to qualify as an improvement. Default = 0.0.
            __verbose (bool): If True, prints a message for each epoch without improvement. Default = True.
        """
        
        self.__patience = patience
        self.__delta = delta
        self.__verbose = verbose
        self.__counter = 0
        self.__best_loss = float('inf')
        self.__early_stop = False

    def __call__(self, val_loss):
        """
        Checks the validation loss and updates the early stopping status.

        Args:
            val_loss (float): The current validation loss.
        """
        
        if val_loss < self.__best_loss - self.__delta:
            self.__best_loss = val_loss
            self.__counter = 0
        else:
            self.__counter += 1
            if self.__verbose:
                print(f"No improvement: {self.__counter}/{self.__patience}")
            if self.__counter >= self.__patience:
                self.__early_stop = True
    
    @property
    def early_stop(self):
        return self.__early_stop
    
    @property
    def patience(self):
        return self.__patience
