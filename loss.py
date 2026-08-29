from .engine import Scalar

def MSELoss(predictions,targets):
    loss=Scalar(0)
    for pred,target in zip(predictions,targets):
        if not isinstance(target, Scalar):
            target=Scalar(target)
        loss+=(pred-target)**2
    return loss/len(predictions)


def BCELoss(predictions,targets):
    loss=Scalar(0)
    for pred,target in zip(predictions,targets):
        if not isinstance(target,Scalar):
            target=Scalar(target)
        loss-=(target*pred.log()+(Scalar(1)-target)*(Scalar(1)-pred).log())
    return loss/len(predictions)


def CELoss(predictions,targets):
    loss=Scalar(0)
    for logits,target in zip(predictions,targets):
        if isinstance(target,Scalar):
            target=int(target.data)
        else:
            target=int(target)
        exps=[(x).exp() for x in logits]
        total=sum(exps)
        probs=[e/total for e in exps]
        loss+=-probs[target].log()
    return loss/len(predictions)
