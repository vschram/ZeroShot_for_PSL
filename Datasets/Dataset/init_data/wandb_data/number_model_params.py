import numpy as np
from collections import OrderedDict

def estimate_model_size_nanogpt(n_layer, n_embd, block_size=1024, vocab_size=50257):
    """ estimates the number of parameters in the model"""
    out = OrderedDict()
    # here, embeddings are included into the final count
    # Code taken from: https://github.com/karpathy/nanoGPT/blob/master/transformer_sizing.ipynb
    # token and position embeddings
    out['emebedding/position'] = n_embd * block_size
    out['embedding/token'] = n_embd * vocab_size
    out['embedding'] = out['emebedding/position'] + out['embedding/token']

    # attention blocks
    out['attention/ln'] = n_embd  # note, bias=False in our LN
    out['attention/kqv'] = n_embd * 3 * n_embd
    out['attention/proj'] = n_embd ** 2
    out['attention'] = out['attention/ln'] + out['attention/kqv'] + out['attention/proj']

    # MLP blocks
    ffw_size = 4 * n_embd  # feed forward size
    out['mlp/ln'] = n_embd
    out['mlp/ffw'] = n_embd * ffw_size
    out['mlp/proj'] = ffw_size * n_embd
    out['mlp'] = out['mlp/ln'] + out['mlp/ffw'] + out['mlp/proj']

    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = n_layer * out['block']
    out['ln_f'] = n_embd  # final layernorm
    out['dense'] = 0  # 0 because of parameter sharing. This layer uses the weights from the embedding layer

    # total
    out['total'] = out['embedding'] + out['transformer'] + out['ln_f'] + out['dense']

    total_number_of_parameters = out['total']
    return total_number_of_parameters

total_number=estimate_model_size_nanogpt(n_layer=8, n_embd=512, block_size=1024, vocab_size=50257)
print(total_number)