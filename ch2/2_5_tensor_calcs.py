import torch

def clamp_test():
    a=torch.arange(0,8,1).reshape_as(torch.Tensor(2,4))
    b=a.clamp(min=3,max=5)

    print("初始张量为:\n",a,"\n经过[3,5]为阈值的剪裁后，变成:\n",b)

def topk_test():
    a=torch.arange(0,27).reshape(3,3,3)
    print("初始张量a:\n",a)

    t2,t2p=torch.topk(a,2,2,True)

    print("\na沿最后一个维度的每一列的最大的两个值为:\n",t2,"\n它们在对应列（切片）中的位置分别是:\n",t2p)

def kthval_test():
    a=torch.randperm(27).reshape(3,3,3)
    print("张量a:\n",a)

    max,_=torch.max(a,2,False)
    print("\n用max得到a的第二维度最大值张量为:\n",max)

    min3,_=torch.kthvalue(a,3,2,False)
    print("\n用kthval得到a的第二维度最大值张量为:\n",min3)

    print("\n当参数合适时，max和kthvalue的行为是一致的:",torch.equal(min3,max))


if __name__ == '__main__':
    #clamp_test()
    #topk_test()
    kthval_test()