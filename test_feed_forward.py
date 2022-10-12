import network as n


def feed_forward_manually(model, x):
    x1, x2 = x[0], x[1]
    w1, w2, w3 = model.layers[0].weights[0]
    w4, w5, w6 = model.layers[0].weights[1]
    bias_layer0 = model.layers[0].bias[0]

    h1 = n.logit_to_prob(x1*w1 + x2*w4 + bias_layer0)
    h2 = n.logit_to_prob(x1*w2 + x2*w5 + bias_layer0)
    h3 = n.logit_to_prob(x1*w3 + x2*w6 + bias_layer0)

    
    w7, w8 = model.layers[1].weights[0]
    w9, w10 = model.layers[1].weights[1]
    w11, w12 = model.layers[1].weights[2]
    bias_layer1 = model.layers[1].bias[0]

    h4 = n.logit_to_prob(h1*w7 + h2*w9 + h3*w11 + bias_layer1)
    h5 = n.logit_to_prob(h1*w8 + h2*w10 + h3*w12 + bias_layer1)
    
    w13 = model.layers[2].weights[0][0]
    w14 = model.layers[2].weights[1][0]
    bias_layer2 = model.layers[2].bias[0]
    
    y_prob = n.logit_to_prob(h4*w13 + h5*w14 + bias_layer2)
    return {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "h5": h5,
        "y_prob": y_prob,
            }
