"""Keep Adam moments on CPU between native, one-parameter GPU updates."""
from collections import defaultdict

import torch


class CPUOffloadedAdamW(torch.optim.AdamW):
    """Use PyTorch's single-tensor AdamW, staging only one parameter's state.

    Parameters and gradients stay on their original device and in FP32. The
    moments return to CPU before the next parameter is updated. Parameter
    groups and checkpoint state use the ordinary AdamW representation.
    """

    def __init__(self, params, **options):
        if options.get('foreach') is not False:
            raise ValueError('CPU moment offload requires optimizer_foreach=false')
        super().__init__(params, **options)
        self.peak_active_moment_bytes = 0
        self._validate_groups()

    def _validate_groups(self):
        devices = set()
        for group in self.param_groups:
            if (group['foreach'] is not False or group.get('fused') or
                    group.get('capturable') or group.get('differentiable')):
                raise ValueError('CPU moment offload requires ordinary single-tensor AdamW')
            for parameter in group['params']:
                if parameter.dtype != torch.float32 or parameter.device.type not in {'cpu', 'cuda'}:
                    raise ValueError('CPU moment offload requires full FP32 parameters')
                devices.add(parameter.device)
        if len(devices) != 1:
            raise ValueError('CPU moment offload requires one parameter device')

    @torch.no_grad()
    def step(self, closure=None):
        self._validate_groups()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = [(group, group['params']) for group in self.param_groups]
        try:
            for group, _ in groups:
                group['params'] = []
            for group, parameters in groups:
                for parameter in parameters:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    for key in list(state):
                        if key != 'step' and torch.is_tensor(state[key]):
                            state[key] = state[key].to(parameter.device)
                    group['params'] = [parameter]
                    try:
                        # Independent per-parameter operations, unchanged
                        # native AdamW arithmetic and per-group learning rates.
                        super().step()
                        size = sum(value.numel() * value.element_size()
                                   for key, value in state.items()
                                   if key != 'step' and torch.is_tensor(value))
                        self.peak_active_moment_bytes = max(self.peak_active_moment_bytes, size)
                    finally:
                        for key in list(state):
                            if torch.is_tensor(state[key]):
                                state[key] = state[key].to('cpu')
                        group['params'] = []
        finally:
            for group, parameters in groups:
                group['params'] = parameters
        return loss

    def load_state_dict(self, state_dict):
        # Ordinary Optimizer loading casts moments to each parameter's device.
        # CPU proxies avoid materializing *all* moments on GPU during resume.
        groups, old_state = self.param_groups, self.state
        if len(state_dict['param_groups']) != len(groups):
            raise ValueError('Offloaded Adam checkpoint parameter groups differ')
        identifiers = []
        for group, saved_group in zip(groups, state_dict['param_groups']):
            if len(group['params']) != len(saved_group['params']):
                raise ValueError('Offloaded Adam checkpoint parameter counts differ')
            identifiers.extend(saved_group['params'])
            for parameter, index in zip(group['params'], saved_group['params']):
                state = state_dict['state'].get(index, {})
                if state and not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys():
                    raise ValueError('Offloaded Adam checkpoint moments are incomplete')
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if key in state and (state[key].dtype != torch.float32 or
                                         state[key].shape != parameter.shape):
                        raise ValueError('Offloaded Adam moments must match FP32 parameter shapes on CPU')
        if len(set(identifiers)) != len(identifiers) or not set(state_dict['state']) <= set(identifiers):
            raise ValueError('Offloaded Adam checkpoint state ownership differs')
        proxies = [[torch.empty(0, dtype=parameter.dtype, device='cpu')
                    for parameter in group['params']] for group in groups]
        self.param_groups = [dict(group, params=parameters)
                             for group, parameters in zip(groups, proxies)]
        try:
            super().load_state_dict(state_dict)
            restored = defaultdict(dict)
            for group, proxy_parameters in zip(groups, proxies):
                for parameter, proxy in zip(group['params'], proxy_parameters):
                    state = self.state.get(proxy, {})
                    if state:
                        for key in ('exp_avg', 'exp_avg_sq'):
                            value = state[key]
                            if (value.device.type != 'cpu' or value.dtype != torch.float32 or
                                    value.shape != parameter.shape):
                                raise ValueError('Offloaded Adam moments must match FP32 parameter shapes on CPU')
                    if state:
                        restored[parameter] = state
            for actual, original in zip(self.param_groups, groups):
                actual['params'] = original['params']
            self.state = restored
            self._validate_groups()
        except BaseException:
            self.param_groups, self.state = groups, old_state
            raise

    def placement_summary(self):
        tensors = [value for state in self.state.values() for value in state.values()
                   if torch.is_tensor(value)]
        return dict(parameters_with_state=len(self.state), state_tensors=len(tensors),
                    parameter_device=str(self.param_groups[0]['params'][0].device),
                    all_states_on_CPU=all(value.device.type == 'cpu' for value in tensors),
                    all_moments_FP32=all(value.dtype == torch.float32
                        for state in self.state.values() for key, value in state.items()
                        if key != 'step' and torch.is_tensor(value)),
                    total_state_bytes=sum(value.numel() * value.element_size() for value in tensors),
                    peak_active_moment_bytes=self.peak_active_moment_bytes)
