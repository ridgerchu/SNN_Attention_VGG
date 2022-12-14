from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import neuron, monitor, base


def stdp_linear_single_step(
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if trace_pre is None:
        trace_pre = 0.

    if trace_post is None:
        trace_post = 0.

    weight = fc.weight.data
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    # shape = [N, C_in]
    trace_post = trace_post - trace_post / tau_post + out_spike
    # shape = [N, C_out]

    # [N, out, in] -> [out, in]
    delta_w_pre = - (f_pre(weight) * 
                (trace_post.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0))
    delta_w_post = (f_post(weight) * 
                (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0))
    return trace_pre, trace_post, delta_w_pre + delta_w_post


def stdp_conv2d_single_step(
    conv: nn.Conv2d, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[torch.Tensor, None], trace_post: Union[torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if conv.dilation != (1, 1):
        raise NotImplementedError(
            'STDP with dilation != 1 for Conv2d has not been implemented!'
        )
    if conv.groups != 1:
        raise NotImplementedError(
            'STDP with groups != 1 for Conv2d has not been implemented!'
        )

    stride_h = conv.stride[0]
    stride_w = conv.stride[1]

    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice,
                mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))

    if trace_pre is None:
        trace_pre = torch.zeros(
            size = [conv.weight.shape[2], conv.weight.shape[3],
            in_spike.shape[0], in_spike.shape[1],
            out_spike.shape[2], out_spike.shape[3]], 
            device=in_spike.device, dtype=in_spike.dtype
        )

    if trace_post is None:
        trace_post = torch.zeros(
            [conv.weight.shape[2], conv.weight.shape[3], *out_spike.shape],
            device = in_spike.device, dtype = in_spike.dtype
        )

    trace_pre = trace_pre - trace_pre / tau_pre + pre_spike
    trace_post = trace_post - trace_post / tau_post + post_spike

    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w

            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]
            # pre_spike.shape = [N, C_in, h_out, w_out]
            post_spike = out_spike
            # post_spike.shape = [N, C_out, h_out, h_out]
            weight = conv.weight.data[:, :, h, w]
            # weight.shape = [C_out, C_in]

            tr_pre = trace_pre[:, :, h:h_end:stride_h, w:w_end:stride_w]
            # tr_pre.shape = [N, C_in, h_out, w_out]
            tr_post = trace_post
            # tr_post.shape = [N, C_out, h_out, w_out]

            delta_w_pre = - (f_pre(weight) * 
			    (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1))
                .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4]))
            delta_w_post = f_post(weight) * \
                        (tr_pre.unsqueeze(1) * post_spike.unsqueeze(1))\
                        .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4])
            delta_w[:, :, h, w] += delta_w_pre + delta_w_post

    return trace_pre, trace_post, delta_w


def stdp_multi_step(
    layer: Union[nn.Linear, nn.Conv2d], 
    in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None], 
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x 
):
    weight = layer.weight.data
    delta_w = torch.zeros_like(weight)
    T = in_spike.shape[0]
    stdp_single_step = (stdp_linear_single_step if isinstance(layer, nn.Linear)
        else stdp_conv2d_single_step)

    for t in range(T):
        trace_pre, trace_post, dw = stdp_single_step(
            layer, in_spike[t], out_spike[t], trace_pre, trace_post,
            tau_pre, tau_post, f_pre, f_post
        )
        delta_w += dw

    return trace_pre, trace_post, delta_w


class STDPLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear], sn: neuron.BaseNode,
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        """
        * :ref:`API in English <STDPLearner.__init__-en>`
        .. _STDPLearner.__init__-cn:

        :param step_mode: ???????????????????????? ``synapse`` ??? ``sn`` ???????????????????????????????????????????????????
        :type step_mode: str
        :param synapse: ?????????
        :type synapse: nn.Conv2d or nn.Linear
        :param sn: ??????????????????
        :type sn: neuron.BaseNode
        :param tau_pre: pre???????????????????????????
        :type tau_pre: float
        :param tau_post: post???????????????????????????
        :type tau_post: float
        :param f_pre: pre????????????????????????
        :type f_pre: Callable
        :param f_post: post????????????????????????
        :type f_post: Callable

        STDP??????????????? ``synapse`` ????????????????????? ``pre_spike``???``sn`` ??????????????? ``post_spike``????????? ``pre_spike`` ??? ``post_spike`` ?????? \
        ``trace_pre`` ??? ``trace_post``?????? :math:`tr[t]` ??????????????????????????????

        .. math::

            tr_{pre}[t] = tr_{pre}[t] - \\frac{tr_{pre}[t-1]}{\\tau_{pre}} + s_{pre}[t]

            tr_{post}[t] = tr_{post}[t] -\\frac{tr_{post}[t-1]}{\\tau_{post}} + s_{post}[t]


        ?????? :math:`tr_{pre}, tr_{post}` ????????????????????????????????? ``tau_pre`` ??? ``tau_post`` ???:math:`s_{pre}[t], s_{post}[t]` ??? ``pre_spike`` ??? ``post_spike``???

        ??????pre????????? ``i`` ???post????????? ``j`` ?????????????????? ``w[i][j]`` ???????????????STDP????????????

        .. math::

            \\Delta W[i][j][t] = F_{post}(w[i][j][t]) \\cdot tr_{i}[t] \\cdot s[j][t] - F_{pre}(w[i][j][t]) \\cdot tr_{j}[t] \\cdot s[i][t]

        ?????? :math:`F_{pre}, F_{post}` ???????????? ``f_pre`` ??? ``f_post``???


        ``STDPLearner`` ?????????2????????????????????? ``synapse`` ??????????????? ``pre_spike``???????????? ``sn`` ??????????????? ``post_spike``????????? \
        ``.enable()`` ??? ``.disable()`` ??????????????????????????? ``STDPLearner`` ????????????????????????

        ?????? ``step(on_grad, scale)`` ???????????????STDP?????????????????????????????????????????? ``delta_w``????????????????????????????????? ``delta_w * scale``????????? ``scale = 1.``???

        ?????????????????? ``on_grad=False`` ??? ``step()`` ???????????? ``delta_w * scale``???
        ????????? ``on_grad=True``?????? ``- delta_w * scale`` ???????????? ``weight.grad``????????????????????????????????? :class:`torch.optim.SGD` ??????????????????????????????????????????????????????????????? ``-``????????????????????? ``weight.data += delta_w * scale``?????????????????????????????? ``weight.data -= lr * weight.grad``???
        ?????? ``on_grad=True``???

        ???????????????``STDPLearner`` ???????????????????????????????????? ``trace_pre`` ??? ``trace_post`` ????????????????????????????????????????????????????????????????????? ``.reset()`` ???????????????

        ???????????????

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import neuron, layer, functional, learning
            step_mode = 's'
            lr = 0.001
            tau_pre = 100.
            tau_post = 50.
            T = 8
            N = 4
            C_in = 16
            C_out = 4
            H = 28
            W = 28
            x_seq = torch.rand([T, N, C_in, H, W])
            def f_weight(x):
                return torch.clamp(x, -1, 1.)

            net = nn.Sequential(
                neuron.IFNode(),
                layer.Conv2d(C_in, C_out, kernel_size=5, stride=2, padding=1, bias=False),
                neuron.IFNode()
            )
            functional.set_step_mode(net, step_mode)
            optimizer = torch.optim.SGD(net[1].parameters(), lr=lr, momentum=0.)
            stdp_learner = learning.STDPLearner(step_mode=step_mode, synapse=net[1], sn=net[2], tau_pre=tau_pre, tau_post=tau_post, f_pre=f_weight, f_post=f_weight)

            with torch.no_grad():
                for t in range(T):
                    weight = net[1].weight.data.clone()
                    net(x_seq[t])
                    stdp_learner.step(on_grad=True)
                    optimizer.step()
                    delta_w = net[1].weight.data - weight
                    print(f'delta_w=\\n{delta_w}')

            functional.reset_net(net)
            stdp_learner.reset()

        * :ref:`??????API <STDPLearner.__init__-cn>`
        .. _STDPLearner.__init__-en:

        :param step_mode: the step mode, which should be same with that of ``synapse`` and ``sn``
        :type step_mode: str
        :param synapse: the synapse
        :type synapse: nn.Conv2d or nn.Linear
        :param sn: the spiking neurons layer
        :type sn: neuron.BaseNode
        :param tau_pre: the time constant of trace of pre neurons
        :type tau_pre: float
        :param tau_post: the time constant of trace of post neurons
        :type tau_post: float
        :param f_pre: the weight function for pre neurons
        :type f_pre: Callable
        :param f_post: the weight function for post neurons
        :type f_post: Callable

        The STDP learner. It will regard inputs of ``synapse`` as ``pre_spike`` and outputs of ``sn`` as ``post_spike``, which will be used to generate ``trace_pre`` and ``trace_post``.

        The update of ``trace_pre`` and ``trace_post`` defined as:

        .. math::

            tr_{pre}[t] = tr_{pre}[t] - \\frac{tr_{pre}[t-1]}{\\tau_{pre}} + s_{pre}[t]

            tr_{post}[t] = tr_{post}[t] -\\frac{tr_{post}[t-1]}{\\tau_{post}} + s_{post}[t]

        where :math:`tr_{pre}, tr_{post}` are time constants???which are ``tau_pre`` and ``tau_post``. :math:`s_{pre}[t], s_{post}[t]` are ``pre_spike`` and ``post_spike``.

        For the pre neuron ``i`` and post neuron ``j``, the synapse ``weight[i][j]`` is updated by the STDP learning rule:

        .. math::

            \\Delta W[i][j][t] = F_{post}(w[i][j][t]) \\cdot tr_{i}[t] \\cdot s[j][t] - F_{pre}(w[i][j][t]) \\cdot tr_{j}[t] \\cdot s[i][t]

        where :math:`F_{pre}, F_{post}` are ``f_pre`` and ``f_post``.


        ``STDPLearner`` will use two monitors to record inputs of ``synapse`` as ``pre_spike`` and outputs of ``sn`` as ``post_spike``. We can use ```.enable()``` or ``.disable()`` to start or pause the monitors.

        We can use ``step(on_grad, scale)`` to apply the STDP learning rule and get the update variation ``delta_w``, while the actual update variation is ``delta_w * scale``. We set ``scale = 1.`` as the default value.

        Note that when we set ``on_grad=False``, then ``.step()`` will return ``delta_w * scale``.
        If we set ``on_grad=True``, then ``- delta_w * scale`` will be added in ``weight.grad``, indicating that we can use optimizers like :class:`torch.optim.SGD` to update weights. Note that there is a negative sign ``-`` because we want the operation ``weight.data += delta_w * scale``, but the optimizer will apply ``weight.data -= lr * weight.grad``.
        We set ``on_grad=True`` as the default value.

        Note that ``STDPLearner`` is also stateful because its ``trace_pre`` and ``trace_post`` are stateful. Do not forget to call ``.reset()`` before giving a new sample to the network.

        Codes example:

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import neuron, layer, functional, learning
            step_mode = 's'
            lr = 0.001
            tau_pre = 100.
            tau_post = 50.
            T = 8
            N = 4
            C_in = 16
            C_out = 4
            H = 28
            W = 28
            x_seq = torch.rand([T, N, C_in, H, W])
            def f_weight(x):
                return torch.clamp(x, -1, 1.)

            net = nn.Sequential(
                neuron.IFNode(),
                layer.Conv2d(C_in, C_out, kernel_size=5, stride=2, padding=1, bias=False),
                neuron.IFNode()
            )
            functional.set_step_mode(net, step_mode)
            optimizer = torch.optim.SGD(net[1].parameters(), lr=lr, momentum=0.)
            stdp_learner = learning.STDPLearner(step_mode=step_mode, synapse=net[1], sn=net[2], tau_pre=tau_pre, tau_post=tau_post, f_pre=f_weight, f_post=f_weight)

            with torch.no_grad():
                for t in range(T):
                    weight = net[1].weight.data.clone()
                    net(x_seq[t])
                    stdp_learner.step(on_grad=True)
                    optimizer.step()
                    delta_w = net[1].weight.data - weight
                    print(f'delta_w=\\n{delta_w}')

            functional.reset_net(net)
            stdp_learner.reset()
        """


        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory('trace_pre', None)
        self.register_memory('trace_post', None)

    def reset(self):
        super(STDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        """
        * :ref:`API in English <STDPLearner.disable-en>`
        .. _STDPLearner.disable-cn:

        ????????? ``synapse`` ???????????? ``sn`` ?????????????????????

        * :ref:`??????API <STDPLearner.disable-cn>`
        .. _STDPLearner.disable-en:

        Pause the monitoring of inputs of ``synapse`` and outputs of ``sn``.
        """
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        """
        * :ref:`API in English <STDPLearner.disable-en>`
        .. _STDPLearner.disable-cn:

        ????????? ``synapse`` ???????????? ``sn`` ?????????????????????

        * :ref:`??????API <STDPLearner.disable-cn>`
        .. _STDPLearner.disable-en:

        Enable the monitoring of inputs of ``synapse`` and outputs of ``sn``.
        """
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, on_grad: bool = True, scale: float = 1.):
        """
        * :ref:`API in English <STDPLearner.step-en>`
        .. _STDPLearner.step-cn:

        :param on_grad: ???????????????????????????????????????????????? ``True`` ????????? ``- delta_w * scale`` ?????? ``weight.grad``????????? ``False`` ????????????????????? ``delta_w * scale``
        :type on_grad: bool
        :param scale: ?????????????????????????????????????????????
        :type scale: float
        :return: None or ``delta_w * scale``
        :rtype: None or torch.Tensor

        * :ref:`??????API <STDPLearner.step-cn>`
        .. _STDPLearner.step-en:

        :param on_grad: whether add the update variation on ``weight.grad``. If ``True``, then ``- delta_w * scale`` will be added on ``weight.grad``. If `False`, then this function will return ``delta_w * scale``
        :type on_grad: bool
        :param scale: the scale of ``delta_w``, which acts like the learning rate
        :type scale: float
        :return: None or ``delta_w * scale``
        :rtype: None or torch.Tensor
        """
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Conv2d):
                stdp_f = stdp_conv2d_single_step
            elif isinstance(self.synapse, nn.Linear):
                stdp_f = stdp_linear_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == 'm':
            if (isinstance(self.synapse, nn.Conv2d) or 
                isinstance(self.synapse, nn.Linear)):
                stdp_f = stdp_multi_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)
            out_spike = self.out_spike_monitor.records.pop(0)

            self.trace_pre, self.trace_post, dw = stdp_f(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post, 
                self.tau_pre, self.tau_post,
                self.f_pre, self.f_post
            )
            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w
