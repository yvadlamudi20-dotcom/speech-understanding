import torch

model = torch.nn.Linear(10,2)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(5):
    x = torch.randn(10)
    y = torch.tensor([1.0,0.0])
    
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred,y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Epoch:", epoch, "Loss:", loss.item())
