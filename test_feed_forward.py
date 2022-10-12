import network as n


def feed_forward_manually(model, x):
    x1, x2 = x[0], x[1]
    w1, w2, w3 = model.layers[0].weights[0]
    w4, w5, w6 = model.layers[0].weights[1]
    h1 = n.logit_to_prob(x1*w1 + x2*w4 + 1)
    h2 = n.logit_to_prob(x1*w2 + x2*w5 + 1)
    h3 = n.logit_to_prob(x1*w3 + x2*w6 + 1)

    
    w7, w8 = model.layers[1].weights[0]
    w9, w10 = model.layers[1].weights[1]
    w11, w12 = model.layers[1].weights[2]
    h4 = n.logit_to_prob(h1*w7 + h2*w9 + h3*w11 + 1)
    h5 = n.logit_to_prob(h1*w8 + h1*w10 + h3*w12 + 1)
    
    w13 = model.layers[2].weights[0][0]
    w14 = model.layers[2].weights[1][0]
    
    y_prob = n.logit_to_prob(h4*w13 + h5*w14 + 1)
    return y_prob
