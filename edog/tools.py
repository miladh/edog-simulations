import quantities as pq
import numpy as np
import warnings

import pylgn
import pylgn.kernels as kernel
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl


def parse_parameters(filename):
    """
    Parse parameters from parameter file.

    Parameters
    ----------
    filename : string
            yaml filename

    Returns
    -------
    out : dict
        Dictionary with all parameters

    """
    import yaml
    from operator import itemgetter

    with open(filename, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    unpacked_params = _unpack_params(params)

    return unpacked_params


def _unpack_params(params):
    """
    Unpacks parameters
    """
    unpacked_params = {}

    for key, value in params.items():
        if isinstance(value, dict):
            if all(k in value for k in ("value", "unit")):
                unpacked_params[key] = pq.Quantity(value["value"], value["unit"])
            elif all(k in value for k in ("start", "stop", "count")):
                unpacked_params[key] = np.linspace(value["start"], value["stop"], value["count"])
                if "unit" in value:
                    unpacked_params[key] = np.linspace(value["start"], value["stop"], value["count"]) * pq.Quantity(1, value["unit"])
            else:
                unpacked_params[key] = _unpack_params(value)
        else:
            unpacked_params[key] = value
    return unpacked_params


def get_neuron(name, network):
    """
    Returns a specific pylgn.Neuron

    Parameters
    ----------
    name : string
         Name of the neuron

    network : pylgn.Network

    Returns
    -------
    out : pylgn.Neuron

    """
    neuron = [neuron for neuron in network.neurons if type(neuron).__name__ == name]
    if not neuron:
        raise NameError("neuron not found in network", name)
    elif len(neuron) > 1 and name == "Relay":
        raise ValueError("more than one Relay cell found in network")
    return neuron


def create_spatial_network(nt, nr, dt, dr,                              # integrator
                           A_g=0, a_g=1*pq.deg, B_g=0, b_g=1*pq.deg,    # Wg
                           A_rg=0, a_rg=1*pq.deg, w_rg=0,               # Krg
                           A_rig=0, a_rig=1*pq.deg, w_rig=0,            # Krig
                           A_rc_ex=0, a_rc_ex=1*pq.deg, w_rc_ex=0,      # Krc_ex
                           A_rc_in=0, a_rc_in=1*pq.deg, w_rc_in=0):     # Krc_in

    """
    Creates lgn network, with all temporal kernels as delta functions.

    Parameters
    ----------
    in : Network parameters. With default kernel parameters
         none of the neurons are connected.

    Returns
    -------
    out : pylgn.Network

    """

    network = pylgn.Network()

    integrator = network.create_integrator(nt=nt, nr=nr, dt=dt, dr=dr)

    ganglion = network.create_ganglion_cell()
    relay = network.create_relay_cell()
    cortical = network.create_cortical_cell()

    delta_t = tpl.create_delta_ft()

    Wg_r = spl.create_dog_ft(A=A_g, a=a_g, B=B_g, b=b_g)
    Krg_r = spl.create_gauss_ft(A=A_rg, a=a_rg)
    Krig_r = spl.create_gauss_ft(A=A_rig, a=a_rig)
    Krc_ex_r = spl.create_gauss_ft(A=A_rc_ex, a=a_rc_ex)
    Krc_in_r = spl.create_gauss_ft(A=A_rc_in, a=a_rc_in)
    Kcr_r = spl.create_delta_ft()

    ganglion.set_kernel((Wg_r, delta_t))
    network.connect(ganglion, relay, (Krg_r, delta_t), weight=w_rg)
    network.connect(ganglion, relay, (Krig_r, delta_t), weight=w_rig)
    network.connect(cortical, relay, (Krc_ex_r, delta_t), weight=w_rc_ex)
    network.connect(cortical, relay, (Krc_in_r, delta_t), weight=w_rc_in)
    network.connect(relay, cortical, (Kcr_r, delta_t), weight=1)

    return network


def create_spatiotemporal_network(nt, nr, dt, dr,                                 # integrator
                                  A_g=0, a_g=1*pq.deg, B_g=0, b_g=1*pq.deg,       # Wg_r
                                  phase=43*pq.ms, damping=0.38, delay_g=0*pq.ms,  # Wg_t
                                  w_rg=0, A_rg=0, a_rg=1*pq.deg,                  # Krg_r
                                  tau_rg=0*pq.ms, delay_rg=0*pq.ms,               # Krg_t
                                  w_rig=0, A_rig=0, a_rig=1*pq.deg,               # Krig_r
                                  tau_rig=0*pq.ms, delay_rig=0*pq.ms,             # Krig_t
                                  w_rc_ex=0, A_rc_ex=0, a_rc_ex=1*pq.deg,         # Krc_ex_r
                                  tau_rc_ex=0*pq.ms, delay_rc_ex=0*pq.ms,         # Krc_ex_t
                                  w_rc_in=0, A_rc_in=0, a_rc_in=1*pq.deg,         # Krc_in_r
                                  tau_rc_in=0*pq.ms, delay_rc_in=0*pq.ms):        # Krc_in_t
    """
    Creates lgn network

    Parameters
    ----------
    in : Network parameters. With default kernel parameters
         none of the neurons are connected.

    Returns
    -------
    out : pylgn.Network

    """
    network = pylgn.Network()

    integrator = network.create_integrator(nt=nt, nr=nr, dt=dt, dr=dr)

    ganglion = network.create_ganglion_cell()
    relay = network.create_relay_cell()
    cortical = network.create_cortical_cell()

    Wg_r = spl.create_dog_ft(A=A_g, a=a_g, B=B_g, b=b_g)
    Wg_t = tpl.create_biphasic_ft(phase=phase, damping=damping, delay=delay_g)

    Krg_r = spl.create_gauss_ft(A=A_rg, a=a_rg)
    if tau_rg == 0*pq.ms:
        Krg_t = tpl.create_delta_ft(delay=delay_rg)
    else:
        Krg_t = tpl.create_exp_decay_ft(tau=tau_rg, delay=delay_rg)

    Krig_r = spl.create_gauss_ft(A=A_rig, a=a_rig)
    if tau_rig == 0*pq.ms:
        Krig_t = tpl.create_delta_ft(delay=delay_rig)
    else:
        Krig_t = tpl.create_exp_decay_ft(tau=tau_rig, delay=delay_rig)

    Krc_ex_r = spl.create_gauss_ft(A=A_rc_ex, a=a_rc_ex)
    if tau_rc_ex == 0*pq.ms:
        Krc_ex_t = tpl.create_delta_ft(delay=delay_rc_ex)
    else:
        Krc_ex_t = tpl.create_exp_decay_ft(tau=tau_rc_ex, delay=delay_rc_ex)

    Krc_in_r = spl.create_gauss_ft(A=A_rc_in, a=a_rc_in)
    if tau_rc_in == 0*pq.ms:
        Krc_in_t = tpl.create_delta_ft(delay=delay_rc_in)
    else:
        Krc_in_t = tpl.create_exp_decay_ft(tau=tau_rc_in, delay=delay_rc_in)

    Kcr_r = spl.create_delta_ft()
    Kcr_t = tpl.create_delta_ft()

    ganglion.set_kernel((Wg_r, Wg_t))
    network.connect(ganglion, relay, (Krg_r, Krg_t), weight=w_rg)
    network.connect(ganglion, relay, (Krig_r, Krig_t), weight=w_rig)
    network.connect(cortical, relay, (Krc_ex_r, Krc_ex_t), weight=w_rc_ex)
    network.connect(cortical, relay, (Krc_in_r, Krc_in_t), weight=w_rc_in)
    network.connect(relay, cortical, (Kcr_r, Kcr_t), weight=1)

    return network


def spatiotemporal_size_tuning_flash(network, patch_diameter,
                                     delay=0*pq.ms, duration=500*pq.ms):
    """
    Computes the spatiotemporal size tuning curve for
    flashing spot stimulus.

    Parameters
    ----------
    network : pylgn.Network
            lgn network

    patch_diameter : quantity array
        patch sizes

    delay : quantity scalar
         stimulus onset

    duration : quantity scalar
            stimulus duration

    Returns
    -------
    out : quantity array
        each column is the temporal response of the center neuron
        for one specific spot size.
    """

    responses = np.zeros([network.integrator.Nt, len(patch_diameter)]) / pq.s

    for i, d in enumerate(patch_diameter):
        stimulus = pylgn.stimulus.create_flashing_spot(patch_diameter=d,
                                                       delay=delay, duration=duration)
        network.set_stimulus(stimulus, compute_fft=True)
        [relay] = get_neuron("Relay", network)
        network.compute_response(relay, recompute_ft=True)
        responses[:, i] = relay.center_response

    return responses


def spatiotemporal_size_tuning(network, angular_freq,
                               wavenumber, patch_diameter):
    """
    Computes the spatiotemporal size tuning curve for
    patch grating stimulus.

    Parameters
    ----------
    network : pylgn.Network
            lgn network

    angular_freq : quantity scalar
                 angular frequency

    wavenumber : quantity scalar
                 wavenumber

    patch_diameter : quantity array
                   patch sizes
    Returns
    -------
    out : quantity array
        each column is the temporal response of the center neuron
        for one specific patch size.

    """
    responses = np.zeros([network.integrator.Nt, len(patch_diameter)]) / pq.s

    for i, d in enumerate(patch_diameter):
        stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=angular_freq,
                                                          wavenumber=wavenumber,
                                                          patch_diameter=d)
        network.set_stimulus(stimulus)
        [relay] = get_neuron("Relay", network)
        network.compute_response(relay, recompute_ft=True)
        responses[:, i] = relay.center_response

    return responses


def spatiotemporal_wavenumber_tuning(network, angular_freq,
                                     wavenumber, patch_diameter):
    """
    Computes the spatiotemporal wavenumber tuning for
    patch grating stimulus.

    Parameters
    ----------
    network : pylgn.Network
            lgn network

    angular_freq : quantity scalar
                 angular frequency

    wavenumber : quantity array
                 wavenumber

    patch_diameter : quantity scalar
                   patch sizes
    Returns
    -------
    out : quantity array
        each column is the temporal response of the center neuron
        for one specific wavenumber.

    """
    responses = np.zeros([network.integrator.Nt, len(wavenumber)]) / pq.s
    for i, kd in enumerate(wavenumber):
        stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=angular_freq,
                                                          wavenumber=kd,
                                                          patch_diameter=patch_diameter)
        network.set_stimulus(stimulus)
        [relay] = get_neuron("Relay", network)
        network.compute_response(relay, recompute_ft=True)
        responses[:, i] = relay.center_response

    return responses


def _find_sign_change(a):
    """
    Detects a sign change in elements of an array.
    Zero is also considered as a sign shift.

    Parameters
    ----------
    a : array

    Returns
    -------
    out : array
         array where 1s are where a sign change occured, 0 otherwise.

    """
    a_sign = np.sign(a)
    sz = a_sign == 0
    signchange = ((np.roll(a_sign, 1) - a_sign) != 0).astype(int)
    signchange[0] = 0

    return signchange


def spatial_irf_params(irf, positions):
    """
    Finds spatial irf params:
        - center excitation
        - surround inhibition
        - center size

    Parameters
    ----------
    irf : quantity array
        sptial irf (2d)
    positions : quantity array
              spatial coordinates

    Returns
    -------
    cen_ex : quantity array
        center excitation
    sur_in : quantity array
        surround inhibition
    cen_size : quantity array
        center size
    """
    cen_ex = irf.max()
    sur_in = irf.min()

    [Nr] = positions.shape
    signchanges = _find_sign_change(irf[int(Nr/2), int(Nr/2):].magnitude)

    try:
        first_signchange = np.where(signchanges == 1)[0][0]
        cen_size = positions[int(Nr/2) + first_signchange]
    except IndexError:
        cen_size = np.nan

    return cen_ex, sur_in, cen_size


def temporal_irf_params(irf, times):
    '''
    Finds temporal irf params:
        - t_peak
        - biphasic index

    Parameters
    ----------
    irf : quantity array
        temporal irf (1d)
    times : quantity array
              time array

    Returns
    -------
    t_peak : quantity scalar
        peak response latency

    I_bp: float
        biphasic index
    '''
    t_peak = times[np.argmax(irf)]

    if np.argmax(irf) < np.argmin(irf):
        A = irf.max()
        B = irf.min()
        I_bp = abs(B / A)
    else:
        warnings.warn("could not calculate biphasic index, argmax, argmin: ",
                      np.argmax(irf), np.argmin(irf))
        I_bp = None

    return t_peak, I_bp


def rf_center_size(tuning, patch_diameter):
    '''
    Finds receptive field center size from size tuning curve.

    Parameters
    ----------
    tuning : quantity array
           size uning curve
    patch_diameter : quantity array
            stimulus patch diameter

    Returns
    -------
    out : quantity scalar
        center size
    '''

    center_idx = int(np.where(tuning == tuning.max())[0][0])
    center_size = patch_diameter[center_idx]

    return center_size


def compute_suppression_index(tuning):
    '''
    Computes the suppresion index from the size tuning curve.

    Parameters
    ----------
    tuning : quantity array
             size uning curve
    Returns
    -------
    out : quantity scalar
        suppresion index
    '''
    supp_index = (tuning.max() - tuning[-1]) / tuning.max()
    return supp_index


def compute_autocorrelation(x):
    '''
    computes the autocorrelation function:
    http://stackoverflow.com/q/14297012/190597
    https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    Parameters
    ----------
    x : quantity array

    Returns
    -------
    out : array
        autocorrelation estimation
    '''

    n = len(x)
    corr = np.correlate(x-x.mean(), x-x.mean(), mode='full')[-n:]
    autocorr = corr / (x.var() * (np.arange(n, 0, -1)))

    return autocorr
