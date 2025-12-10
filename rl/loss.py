import torch

from tensordict import TensorDictBase
from torch import Tensor
from torchrl.objectives.sac import DiscreteSACLoss


class NegamaxDiscreteSACLoss(DiscreteSACLoss):
    def _compute_target(self, tensordict) -> Tensor:
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            next_tensordict = tensordict.get("next").clone(False)

            if self.skip_done_states:
                done = next_tensordict.get(self.tensor_keys.done)
                if done is not None and done.any():
                    next_tensordict_select = next_tensordict[~done.squeeze(-1)]
                else:
                    next_tensordict_select = next_tensordict

                # get probs and log probs for actions computed from "next"
                with self.actor_network_params.to_module(self.actor_network):
                    next_dist = self.actor_network.get_dist(next_tensordict_select)
                next_prob = next_dist.probs
                next_log_prob = torch.log(torch.where(next_prob == 0, 1e-8, next_prob))

                # get q-values for all actions
                next_tensordict_expand = self._vmap_qnetworkN0(
                    next_tensordict_select, self.target_qvalue_network_params
                )
                next_action_value = next_tensordict_expand.get(
                    self.tensor_keys.action_value
                )

                # like in continuous SAC, we take the minimum of the value ensemble and subtract the entropy term
                next_state_value = (
                    next_action_value.min(0)[0] - self._alpha * next_log_prob
                )
                # unlike in continuous SAC, we can compute the exact expectation over all discrete actions
                next_state_value = (next_prob * next_state_value).sum(-1).unsqueeze(-1)
                if next_tensordict_select is not next_tensordict:
                    mask = ~done
                    next_state_value = next_state_value.new_zeros(
                        mask.shape
                    ).masked_scatter_(mask, next_state_value)
            else:
                # get probs and log probs for actions computed from "next"
                with self.actor_network_params.to_module(self.actor_network):
                    next_dist = self.actor_network.get_dist(next_tensordict)
                next_prob = next_dist.probs
                next_log_prob = torch.log(torch.where(next_prob == 0, 1e-8, next_prob))

                # get q-values for all actions
                next_tensordict_expand = self._vmap_qnetworkN0(
                    next_tensordict, self.target_qvalue_network_params
                )
                next_action_value = next_tensordict_expand.get(
                    self.tensor_keys.action_value
                )
                # like in continuous SAC, we take the minimum of the value ensemble and subtract the entropy term
                next_state_value = (
                    next_action_value.min(0)[0] - self._alpha * next_log_prob
                )
                # unlike in continuous SAC, we can compute the exact expectation over all discrete actions
                next_state_value = (next_prob * next_state_value).sum(-1).unsqueeze(-1)

            next_state_value = -next_state_value  # Negamax adjustment
            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def _actor_loss(
        self, tensordict: TensorDictBase
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # get probs and log probs for actions
        with self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict.clone(False))
        prob = dist.probs
        log_prob = torch.log(torch.where(prob == 0, 1e-8, prob))

        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)

        td_q = self._vmap_qnetworkN0(
            td_q, self._cached_detached_qvalue_params  # should we clone?
        )
        min_q = td_q.get(self.tensor_keys.action_value).min(0)[0]

        if log_prob.shape != min_q.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q.shape}"
            )

        # like in continuous SAC, we take the entropy term and subtract the minimum of the value ensemble
        loss = self._alpha * log_prob - min_q
        # unlike in continuous SAC, we can compute the exact expectation over all discrete actions
        loss = (prob * loss).sum(-1)

        return loss, {"log_prob": (log_prob * prob).sum(-1).detach()}