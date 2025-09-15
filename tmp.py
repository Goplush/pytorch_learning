import torch

a=torch.arange(27.0).reshape(3,3,3)

print("原始张量a为:\n",a)

b=torch.chunk(a,6,2)
print(
    "\n想通过torch.chunk(a,6,2)把a沿着第[2]维拆分成六块，但是实际返回 ",
    b.__len__(),
    " 块，每块的内容为:\n"
)
for i in b:
    print(i,"\n")

c=torch.split(a,2,0)
print(
    "\n想通过torch.split(a,2,0)把a沿着第[0]维拆分成两块，实际返回 ",
    c.__len__(),
    " 块，每块的内容为:\n"
)
for i in c:
    print(i,"\n")