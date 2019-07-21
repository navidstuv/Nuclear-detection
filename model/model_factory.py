from model.all_models import get_spagettiNet_singleHead_multiscale_residual_deep


def getModel(network, cellLoss, marginLoss,input_shape):
    if network == 'spagetti-multiscale-residual':
        return get_spagettiNet_twoHeaded_multiscale_residual(input_shape,cellLoss, marginLoss)
    elif network == 'spagetti-multiscale-residual-deep':
        return get_spagettiNet_twoHeaded_multiscale_residual_deep(input_shape,cellLoss, marginLoss)
    elif network == 'spagetti-multiscale-residual-veryDeep':
        return get_spagettiNet_twoHeaded_multiscale_residual_veryDeep(input_shape,cellLoss, marginLoss)
    elif network == 'spagetti-singleHead-multiscale-residual-deep':
        return get_spagettiNet_singleHead_multiscale_residual_deep(input_shape,cellLoss)
    elif network == 'spagetti-singleHead-multiscale-residual':
        return get_spagettiNet_singleHead_multiscale_residual(input_shape,cellLoss)
    elif network == 'spagetti-multiscale-deep':
        return get_spagettiNet_twoHeaded_multiscale_deep(input_shape,cellLoss, marginLoss)
    elif network == 'unet-multiscale-residual':
        return get_UNET_twoHead_multiscale_residual(input_shape,cellLoss, marginLoss)
    elif network == 'unet-multiscale-residual-deep':
        return get_UNET_twoHead_multiscale_residual_deep(input_shape,cellLoss, marginLoss)
    elif network == 'unet-multiscale-residual-shallow':
        return get_UNET_twoHead_multiscale_residual_shallow(input_shape,cellLoss, marginLoss)
    else:
        raise ValueError('unknown network ' + network)