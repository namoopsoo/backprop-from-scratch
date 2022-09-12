import plot
import network as n


def train_and_analysis(data, parameters):

    model = n.initialize_model(parameters)

    (metrics, model, artifacts, Y_prob) = n.train_network(
        data,
        model,
        log_loss_every_k_steps=parameters["log_loss_every_k_steps"],
        steps=model.parameters["steps"])

    out_loc = plot.plot_train_and_validation_loss_vec(
        metrics["train"]["loss_vec"],
        metrics["validation"]["loss_vec"]
    )
    print(out_loc)

    out_loc = plot.plot_model_weights_across_rounds(model, artifacts)
    print(out_loc)

    out_loc = plot.plot_simple_historgram(Y_prob, label="Y_prob")
    print(out_loc)

    out_loc = plot.scatter_plot_by_z(data.X_validation, Y_prob, scaled=True) 
    print(out_loc)

    # TODO should automatically probably create this as a joblib metrics bundle, and then I can just load it after creating it . 


    return model, artifacts



