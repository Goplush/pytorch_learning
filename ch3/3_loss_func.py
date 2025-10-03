import torch


def torch_autograd_test():
    # 测试操作是否可微分

    def test_operation(x:torch.Tensor):
        #该方法为需要测试是否是pytorch可微分的方法
        #the operation to be tested
        return x.add(x)

    x = torch.randn(3, 3, requires_grad=True)
    try:
        y = test_operation(x)
        loss = y.sum()
        loss.backward()
        print("可微分！梯度存在:", x.grad is not None)
    except:
        print("不可微分！")




if __name__ == '__main__':
    torch_autograd_test()