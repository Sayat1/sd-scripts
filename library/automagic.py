from typing import List
import torch
from library.optimizer_utils import Auto8bitTensor, copy_stochastic, stochastic_grad_accummulation
import random

# from https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py
class Automagic(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-6,  # lr is start lr
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=1e-6,  # amount to bump the lr when adjusting
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        beta2=0.999,
        weight_decay=0.0,
        do_paramiter_swapping=False,
        paramiter_swapping_factor=0.1,
        use_adopt=False,
        beta1=0.9,
        use_orthograd=False,
    ):
        self.lr = lr
        if self.lr > 1e-3:
            print(f"Warning! Start lr is very high: {self.lr}. Forcing to 1e-6. this does not work like prodigy")
            self.lr = 1e-6
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.use_adopt = use_adopt

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "use_adopt": use_adopt,
            "use_orthograd": use_orthograd,
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [lr for group in self.param_groups]

        self.is_stochastic_rounding_accumulation = False

        # setup stochastic grad accum hooks
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(stochastic_grad_accummulation)

        self.do_paramiter_swapping = do_paramiter_swapping
        self.paramiter_swapping_factor = paramiter_swapping_factor
        
        # count total parameters
        self._total_paramiter_size = sum(p.numel() for group in self.param_groups for p in group["params"])
        # pretty print total parameters with comma separation
        print(f"Total training parameters: {self._total_paramiter_size:,}")

        # needs to be enabled to count parameters
        if self.do_paramiter_swapping:
            self.enable_paramiter_swapping(self.paramiter_swapping_factor)

    def enable_paramiter_swapping(self, paramiter_swapping_factor=0.1):
        self.do_paramiter_swapping = True
        self.paramiter_swapping_factor = paramiter_swapping_factor
        # call it an initial time
        self.swap_paramiters()

    def swap_paramiters(self):
        all_params = []
        # deactivate all parameters
        for group in self.param_groups:
            for param in group["params"]:
                param.requires_grad_(False)
                # remove any grad
                param.grad = None
                all_params.append(param)
        # shuffle all parameters
        random.shuffle(all_params)

        # keep activating parameters until we are going to go over the target parameters
        target_paramiters = int(self._total_paramiter_size * self.paramiter_swapping_factor)
        total_paramiters = 0
        for param in all_params:
            total_paramiters += param.numel()
            if total_paramiters >= target_paramiters:
                break
            else:
                param.requires_grad_(True)

    @staticmethod
    def _get_lr(param_group, param_state):
        if "avg_lr" in param_state:
            lr = param_state["avg_lr"]
        else:
            lr = 0.0
        return lr

    def _get_group_lrs(self, group):
        group_lrs = []
        for p in group["params"]:
            p_lr = self._get_lr(group, self.state[p])
            group_lrs.append(p_lr.item() if isinstance(p_lr, torch.Tensor) else p_lr)
        return group_lrs

    def _get_group_lr(self, group):
        group_lrs = self._get_group_lrs(group)
        # return avg
        if len(group_lrs) == 0:
            return [self.lr]
        return [sum(group_lrs) / len(group_lrs), max(group_lrs), min(group_lrs)]

    @staticmethod
    def _rms(tensor):
        # Optimized: sqrt(mean(x^2))
        return tensor.pow(2).mean().sqrt()

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    # automagic manages its own lr
    def get_learning_rates(self):
        lrs = [self._get_group_lr(group) for group in self.param_groups]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs

    def get_avg_learning_rate(self):
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.step_hook()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Pre-calculate common constants
        # 4 / pi approx
        ATAN_CONSTANT = 1.2732395447351627

        for group in self.param_groups:
            # Group-level parameters
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta1_inv = 1.0 - beta1
            beta2_inv = 1.0 - beta2
            
            eps_val = group["eps"]
            if isinstance(eps_val, (tuple, list)):
                eps_val = eps_val[0]
            local_eps = eps_val if eps_val is not None else 1e-8
            use_atan2 = group["eps"] is None
            
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]
            use_orthograd = group.get("use_orthograd", False)
            use_adopt = group.get("use_adopt", self.use_adopt)

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Automagic does not support sparse gradients.")

                state = self.state[p]
                # State Initialization
                if not state:
                    self.initialize_state(p)
                
                state["step"] += 1
                step = state["step"]

                # Handle FP32 master copy or direct pointer
                p_data_fp32 = p if p.dtype == torch.float32 else p.clone().float()

                # Common gradient squared calculation
                update_sq = grad.pow(2).add(local_eps)

                if use_adopt:
                    if step == 1:
                        # ADOPT step 1: only update exp_avg_sq
                        state["exp_avg_sq"].add_(update_sq)
                        continue

                    # ADOPT step > 1
                    de_nom_inv = state["exp_avg_sq"].rsqrt()

                    normed_grad = grad * de_nom_inv

                    # ADOPT-style clipping
                    clip = step**0.25
                    normed_grad.clamp_(-clip, clip)

                    state["exp_avg"].lerp_(normed_grad, beta1_inv)
                    update = state["exp_avg"].clone()

                    if use_atan2:
                        # atan2(update, 1) = atan(update)
                        update = torch.atan(update).mul_(ATAN_CONSTANT)
                else:
                    # Non-ADOPT path
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update_sq, alpha=beta2_inv)

                    if use_atan2:
                        # atan2(grad, sqrt(exp_avg_sq)) = atan(grad / sqrt(exp_avg_sq))
                        update = torch.atan(grad * exp_avg_sq.rsqrt()).mul_(ATAN_CONSTANT)
                    else:
                        update = grad * exp_avg_sq.rsqrt()

                if use_orthograd:
                    self.apply_orthogonal_gradients(p)

                if clip_threshold > 0:
                    update.div_((self._rms(update) / clip_threshold).clamp_(min=1.0))

                # Learning rate mask update
                lr_mask = state["lr_mask"].to(torch.float32)
                last_polarity = state["last_polarity"]
                current_polarity = update > 0
                
                # Sign agreement (optimized: match = (last_polarity == current_polarity))
                match = (last_polarity == current_polarity)
                
                # new_lr calculation
                new_lr = torch.where(match, lr_mask + self.lr_bump, lr_mask - self.lr_bump)
                new_lr.clamp_(min=self.min_lr, max=self.max_lr)

                state["last_polarity"] = current_polarity
                state["lr_mask"] = Auto8bitTensor(new_lr)
                state["avg_lr"] = new_lr.mean()

                # Apply weight decay (per-parameter LR)
                if weight_decay != 0:
                    # p = p * (1 - wd * lr)
                    p_data_fp32.mul_(1.0 - weight_decay * new_lr)

                # Final update: p = p - update * new_lr
                p_data_fp32.addcmul_(update, new_lr, value=-1.0)

                if use_adopt:
                    # ADOPT: update exp_avg_sq at the end
                    state["exp_avg_sq"].mul_(beta2).add_(update_sq, alpha=beta2_inv)

                if p.dtype != torch.float32:
                    # apply stochastic rounding
                    copy_stochastic(p, p_data_fp32)

        return loss

    @torch.no_grad()
    def apply_orthogonal_gradients(self, p, eps: float = 1e-16):
        if p.grad is None or p.grad.is_sparse or torch.is_complex(p):
            return

        w = p.view(-1)
        g = p.grad.view(-1)

        # Ensure float32 for dot products to avoid precision issues
        w_f32 = w.to(torch.float32)
        g_f32 = g.to(torch.float32)

        w_dot_w = torch.dot(w_f32, w_f32)
        if w_dot_w <= 1e-30:
            return

        proj = torch.dot(w_f32, g_f32).div_(w_dot_w.add_(eps))
        g_ortho = g_f32.sub(w_f32, alpha=proj)

        g_norm = g_f32.norm(2)
        g_ortho_norm = g_ortho.norm(2)

        g_ortho_scaled = g_ortho.mul_(g_norm.div_(g_ortho_norm.add_(eps)))

        p.grad.copy_(g_ortho_scaled.view_as(p.grad))

    def initialize_state(self, p):
        state = self.state[p]
        state["step"] = 0

        # store the lr mask
        if "lr_mask" not in state:
            state["lr_mask"] = Auto8bitTensor(torch.full(p.shape, self.lr, device=p.device, dtype=torch.float32))
            
        state["avg_lr"] = torch.mean(state["lr_mask"].to(torch.float32))
        
        if "last_polarity" not in state:
            state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)

        # Unified exp_avg_sq
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

        if self.use_adopt:
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)

    # override the state_dict to save the lr_mask
    def state_dict(self, *args, **kwargs):
        orig_state_dict = super().state_dict(*args, **kwargs)
        # convert the state to quantized tensor to scale and quantized
        new_sace_state = {}
        for p, state in orig_state_dict["state"].items():
            save_state = {k: v for k, v in state.items() if k != "lr_mask"}

            # Check if lr_mask exists in the state before trying to access it
            if "lr_mask" in state:
                save_state["lr_mask"] = state["lr_mask"].state_dict()

            new_sace_state[p] = save_state

        orig_state_dict["state"] = new_sace_state

        return orig_state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Validate that the state_dict is from an Automagic optimizer
        is_valid_automagic_state = False

        # Check if state_dict has the expected structure
        if "state" in state_dict and isinstance(state_dict["state"], dict):
            # Check if at least one state entry has an lr_mask, which is specific to Automagic
            for param_id, param_state in state_dict["state"].items():
                if isinstance(param_state, dict) and "lr_mask" in param_state:
                    is_valid_automagic_state = True
                    break

        if not is_valid_automagic_state:
            # Fallback: if it's a regular state dict, still try to load it
            super().load_state_dict(state_dict)
            return

        # First, call the parent class's load_state_dict to load the basic optimizer state.
        # This will also handle the mapping of parameter IDs and move tensors to the correct device.
        super().load_state_dict(state_dict)

        # Now handle the lr_mask separately, converting it back to Auto8bitTensor
        for i, (p, state) in enumerate(self.state.items()):
            if "lr_mask" in state and isinstance(state["lr_mask"], dict):
                saved_lr_mask = state["lr_mask"]

                # Reconstruct the Auto8bitTensor from its state dict
                try:
                    # Make sure the shapes match
                    if "quantized" in saved_lr_mask and saved_lr_mask["quantized"].shape == p.shape:
                        state["lr_mask"] = Auto8bitTensor(saved_lr_mask)
                    else:
                        print(
                            f"WARNING: Shape mismatch for parameter {i}. "
                            f"Expected {p.shape}, got {saved_lr_mask['quantized'].shape if 'quantized' in saved_lr_mask else 'unknown'}. "
                            f"Initializing new lr_mask."
                        )
                        # Initialize a new lr_mask
                        state["lr_mask"] = Auto8bitTensor(
                            torch.full(p.shape, self.lr, device=p.device, dtype=torch.float32)
                        )
                except Exception as e:
                    print(f"ERROR: Failed to load lr_mask for parameter {i}: {e}")
                    # Initialize a new lr_mask
                    state["lr_mask"] = Auto8bitTensor(
                        torch.full(p.shape, self.lr, device=p.device, dtype=torch.float32)
                    )
