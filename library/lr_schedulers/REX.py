from torch.optim.lr_scheduler import LambdaLR

def lr_lambda_rex(
        scheduler_steps: int,
):
    def lr_lambda(current_step: int):
        # https://arxiv.org/abs/2107.04197
        max_lr = 1
        min_lr = 0
        d = 0.9

        if current_step < scheduler_steps:
            progress = (current_step / scheduler_steps)
            div = (1 - d) + (d * (1 - progress))
            return min_lr + (max_lr - min_lr) * ((1 - progress) / div)
        else:
            return min_lr

    return lr_lambda

class REX(LambdaLR):
    def __init__(self,optimizer,max_steps,last_epoch=-1):
        super().__init__(optimizer, lr_lambda=lr_lambda_rex(max_steps), last_epoch=last_epoch)

