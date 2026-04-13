import numpy as np

def sigmoid(x):                        #this implementation takes care of large negative numbers (large positive ones)
    return np.where(x >= 0, 
        1 / (1 + np.exp(-x)),        
        np.exp(x) / (1 + np.exp(x))) 

class SkipGramNS:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01          # (V, N)
        self.W_prim = np.random.randn(vocab_size, embed_dim) * 0.01     # (V, N)    
        
    def forward_backward(self, centers, positives, negatives):
    
        B = centers.shape[0]            # B   batch size, how many words we train at once
                                        # K   Number of negative samples
                                        # N   Number of  hidden layer dim
        #FETCH EMBEDDINGS
        w_i = self.W[centers]                     # (B, N)      matrix of B input words of length N
        w_o = self.W_prim[positives]              # (B, N)      matrix of B output words od length N, for positive samples
        w_j = self.W_prim[negatives]              # (B, K, N)   tensor of BxK output words of length N, for negative samples
        
        #SCORES
            #positive scores
        pos_scores = np.sum(w_i * w_o, axis=1)       # (B,)     calculate positive scores
        pos_sigmoid = sigmoid(pos_scores)            # (B,)     turn those into (0,1) values
            #negative scores
        neg_scores = np.sum(w_i[:, np.newaxis, :] * w_j, axis=2)    # (B, K)     calculate negative score
        neg_sigmoid = sigmoid(-neg_scores)                          # (B, K)
        
        #LOSS
        loss = - (np.sum(np.log(pos_sigmoid + 1e-10)) + np.sum(np.log(neg_sigmoid + 1e-10)))  #Equation 55
        
        #GRADIENTS
            #positive scores 
        grad_pos = (pos_sigmoid - 1)[:, np.newaxis]     # (B,1)    Equation 57 (upper part)
            #negative scores
        grad_neg = (1 - neg_sigmoid)                    # (B,K)    Equation 57 (lower part)
        
        #w_0 updates
            #pos_scores
        grad_w_o = grad_pos * w_i                                           # (B,N)     Equation 58 
            #neg_scores
        grad_w_j = grad_neg[:, :, np.newaxis] * w_i[:, np.newaxis, :]       # (B,K,N)   Equation 58
        
        #w_i updates
            #pos_scores
        grad_w_i = grad_pos * w_o                                           # (B,N)     Equation 61
            #neg_scores
        grad_w_i += np.sum(grad_neg[:, :, np.newaxis] * w_j, axis=1)        # (B, N)    Equation 61     
                
        return loss, grad_w_i, grad_w_o, grad_w_j
    
    def update(self, centers, positives, negatives, grad_w_i, grad_w_o, grad_w_j, lr):
    
        #SGD
        np.add.at(self.W, centers, -lr * grad_w_i)
        np.add.at(self.W_prim, positives, -lr * grad_w_o)
        np.add.at(self.W_prim, negatives, -lr * grad_w_j)       