import numpy as np
from numpy import NDArray
from ..types import BasalModel
import loss_functions as loss


# ------------------------------------------------------------------------------------------------------------
# ----------------------------------CASCADE OPTIMIZERS------------------------------------------------------
def cascade_sgd_momentum(incoming_x, incoming_t, iterations, this_cascade, error_trace, learning_rate_eta=0.02,
                         momentum_alpha=0.96, batch_size=4):
    '''**********ARGUMENTS**********
    :param incoming_x: input values - incoming data should be standardized
    :param incoming_t: target values - targets should be standardized
    :param iterations: number of updates
    :param this_cascade: passes in cascade assembly object
    :param error_trace: error trace object from network object
    :param learning_rate_eta: learning rate value for nnet weight updates
    :param momentum_alpha: Momentum to roll down gradient
    :param batch_size: split batch as 1/nth of the full dataset at a time
    '''
    n_samples = incoming_x.shape[0]
    # n_outputs = incoming_t.shape[-1]

    # make indexing array, shuffle it and use it to index X and T to create batches
    selector = np.array(np.arange(0, n_samples))
    #batch_size = int((1 / batch_size) * n_samples)

    #velocity = this_cascade.momentum

    for i in range(0, iterations):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]
            # print(f'training on {batch_i} : {batch_i + batch_size}, {this_batch.shape}')
            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            output = this_cascade.forward(_x)

            error = loss.loss_dictionary[this_cascade.loss_func](output, _t)

            d_error = loss.derivative_dictionary[this_cascade.loss_func + '_derivative'](output, _t)
            #TODO: add L2 / weight decay: seperate method / function here -
            #TODO: L2 = costF + gamma / n_samples * sum (abs(weights))
            #add gamma/n * w to der of all weights - leave biases intact.
            delta = d_error / batch_size

            output, gradients, delta = this_cascade.compute_gradients(delta)

            # update V factors in this batch
            this_cascade.momentum = (momentum_alpha * this_cascade.momentum) - (learning_rate_eta * gradients)

            #Update weights
            this_cascade.update_weights(this_cascade.momentum)

            batch_error.append(error)

        error_trace.append(np.mean(batch_error))

# ------------------------------------------------------------------------------------------------------------
# ----------------------------------NETWORK OPTIMIZERS------------------------------------------------------

def network_sgd_momentum(incoming_x, incoming_t,
                         network_object, error_trace, iterations,
                         learning_rate_eta=0.02, momentum_alpha=0.5, batch_size=4):
    '''**********ARGUMENTS**********
    :param incoming_x: input values - incoming data should be standardized
    :param incoming_t: target values - targets should be standardized
    :param iterations: number of updates
    :param network_object: passes in as self from Adaptive_Cascade Object
    :param error_trace: error trace object from network object
    :param learning_rate_eta: learning rate value for nnet weight updates
    :param momentum_alpha: Momentum to roll down gradient
    :param batch_size: split batch as 1/nth of the full dataset at a time
    '''
    n_samples = incoming_x.shape[0]

    selector = np.array(np.arange(0, n_samples))

    for i in range(0, iterations):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]

            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            full_output = network_object.forward(_x)
            error = loss.loss_dictionary[network_object.network[-1].loss_func](full_output, _t)
            d_error = loss.derivative_dictionary[network_object.network[-1].loss_func + '_derivative'](full_output, _t)
            delta = d_error / batch_size

            for c_i, this_cascade in enumerate(reversed(network_object.network)):

                _out, gradients, delta = this_cascade.compute_gradients(delta)

                this_cascade.momentum = -(momentum_alpha * this_cascade.momentum) - (learning_rate_eta * gradients)
                this_cascade.update_weights(this_cascade.momentum)
                delta = d_error / batch_size

            batch_error.append(error)

        error_trace.append(np.mean(batch_error))


LAMBDA_MAX = 1e24
LAMBDA_MIN =  1e-24
# TODO: model with a calculate_gradient() func --
def scaled_conjugate_gradient(model: BasalModel,
                              x_data: NDArray,
                              y_data: NDArray,
                              iterations: int) -> NDArray:
    """
    This implementation is based on Charles Anderson's (Colorado State
    University) SCG method --
    It is a 'destructive' -- weights are adjusted as we take steps down the
    gradient. Use with caution!
    Requres a model object with specific methods implemented, hence the
    BasalModel class.

    Parameters
    ----------
    model : BasalModel
    x_data :
    y_data :
    iterations :

    Returns
    -------

    """
    sigma_zero = 1e-6
    lamb = 1e-6
    lamb_ = 0

    vector = model.get_weights().reshape(-1, 1)
    grad_new, _ = model.calculate_gradients(x_data, y_data)
    grad_new = -1 * grad_new.reshape(-1, 1)
    r_new = grad_new.copy()
    success = True

    for _i in range(iterations):
        r = r_new.copy()
        grad = grad_new.copy()
        mu = grad.T @ grad

        if success:
            success = False
            sigma = sigma_zero / np.sqrt(mu)

            grad_old, _ = model._calculate_gradients(x_data, y_data)
            grad_old = grad_old.reshape(-1, 1)

            # update our model's weights -- (take a step down the gradient)
            model.weights = (vector + (sigma * grad)).reshape(model._weight_shape)
            grad_step, _ = model._calculate_gradients(x_data, y_data)

            step = (grad_old - grad_step.reshape(-1, 1)) / sigma
            delta = grad.T @ step

        # increase the curvature - "reach deeper"
        zeta = lamb - lamb_
        step += zeta * grad
        delta += zeta * mu

        if delta <= 0:
            step += (lamb - 2 * delta / mu) * grad
            lamb_ = 2 * (lamb - delta / mu)
            delta -= lamb * mu
            delta *= -1
            lamb = lamb_

        phi = grad.T @ r
        alpha = phi / delta
        vector_new = vector + alpha * grad
        loss_old = model._calculate_loss(x_data, y_data)

        # update our model's weights -- (take a step down the gradient)
        model.weights = vector_new.copy().reshape(model._weight_shape)
        loss_new = model._calculate_loss(x_data, y_data)

        comparison = 2 * delta * (loss_old - loss_new) / (phi ** 2)

        if comparison >= 0:
            # break condition?
            vector = vector_new.copy()
            loss_old = loss_new

            # update our model's weights -- (take a step down the gradient)
            model.weights = vector_new.copy().reshape(model._weight_shape)
            r_new, _ = model._calculate_gradients(x_data, y_data)
            r_new = -1 * r_new.reshape(-1, 1)
            success = True
            lamb_ = 0

            if _i % model._weight_shape[0] == 0:
                grad_new = r_new
            else:
                beta = ((r_new.T @ r_new) - (r_new.T @ r)) / phi
                # update our model's weights -- (take a step down the gradient)
                grad_new = r_new + beta * grad

            if comparison > 0.75:
                lamb = max(0.5 * lamb, LAMBDA_MIN)
        else:
            lamb_ = lamb

        if comparison < 0.25:
            lamb = min(4 * lamb, LAMBDA_MAX)

    return vector_new.reshape(model._weight_shape)
