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
          

'''
class InverseRegularizer(regularizers.Regularizer):
  def __init__(self, strength):
      self.strength = strength

  def __call__(self, x):
      return -self.strength * (tf.reduce_sum(tf.square(x))+0.01)


class afterEpoch_HPTuning_EarlyStopping(Callback):
    def __init__(self,model, stop_tau=10,patience=10,tol=1e-5,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.old_loss = None
        #self.best_loss = None

        self.all_improv_percs = []

        self.stop_tau = stop_tau
        self.patience = patience
        self.tol = tol
        self.patience_cont = 0

        tanh_layers_positions = np.where([type(l) is TanhProd for l in model.layers])[0]
        print(tanh_layers_positions)
        self.tanh_layers = []
        self.stop_training = [True for _ in range(len(tanh_layers_positions))]
        if len(tanh_layers_positions)>0:
          for i,pos in enumerate(tanh_layers_positions):
            self.tanh_layers.append(model.layers[pos])
            self.stop_training[i] = False

    
    def on_epoch_end(self, epoch, logs=None):
        #update tau
        new_loss = logs["loss"]
        if self.old_loss:
            improv_perc = (self.old_loss-new_loss)/self.old_loss #percentage of improvement (0% = same loss, 50% = half loss, 100% zero loss, -10% = increase of 10%)
            #options: #1-np.exp(1-self.old_loss/new_loss)#np.log(self.old_loss/new_loss)#(self.old_loss-new_loss)/self.old_loss

            self.all_improv_percs.append(improv_perc)

            #Change tanh_layer tau
            for tanh_layer in self.tanh_layers:
              min_weight = K.expand_dims(K.max([K.min(K.abs(K.get_value(tanh_layer.kernel))),K.constant(1e-5)]))
              K.set_value(tanh_layer.tau,K.expand_dims(K.max([K.get_value(tanh_layer.tau),1/min_weight]))*(1+improv_perc))

            #check patience
            if improv_perc<=self.tol:
              self.patience_cont +=1

              #if over_patience
              if self.patience_cont > self.patience:
                #print("PATIENCE OVER")
                for i,tanh_layer in enumerate(self.tanh_layers):
                  min_weight = K.expand_dims(K.max([K.min(K.abs(K.get_value(tanh_layer.kernel))),K.constant(1e-5)]))
                  app = self.stop_tau*min_weight-tanh_layer.tau
                  #print("1",app)
                  if app<=0:
                    self.stop_training[i] = True
                  else:
                    #print("CHANGE TAU")
                    print(1/min_weight)
                    K.set_value(tanh_layer.tau,K.expand_dims(K.max([K.get_value(tanh_layer.tau),1/min_weight]))*1.01) #increase by 1%?
              
                if np.all(self.stop_training):
                  self.model.stop_training = True
            else:
              #print("NEW PATIENCE")
              #print()
              self.patience_cont = 0
              #self.best_loss = new_loss

        self.old_loss = new_loss
        #self.best_loss = new_loss

        #and stop_tau:
'''

