import numpy as np

from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers

import tensorflow as tf

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, patience = 10, tol = 1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.old_loss = None
        self.best_loss = np.inf

        self.patience = patience
        self.tol = tol
        self.patience_cont = 0
    
    def on_epoch_end(self, epoch, logs=None):
        new_loss = logs["loss"]
        if (self.best_loss-new_loss) < self.tol:
          #check patience
          self.patience_cont +=1

          #if over_patience
          if self.patience_cont > self.patience:
            #print("PATIENCE OVER")
            self.model.stop_training = True
        else:
          #print("NEW PATIENCE")
          #print()
          self.patience_cont = 0
          self.best_loss = new_loss

        #self.old_loss = new_loss