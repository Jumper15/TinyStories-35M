import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
import torch
import wandb

def model_pretraining(model, dataloader, iterations, batch_size, seq_len, lr):
    scaler = amp.GradScaler()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        lr, 
        total_steps=iterations, 
        pct_start=0.1, 
        anneal_strategy='cos', 
        final_div_factor=100.0 # iterations must be divisible by 100
    )
    
    print("starting training")
    
    
    for epoch in range(iterations):
        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f"model_state/TinyStories_{epoch}.pt")
        X, y = dataloader.load(batch_size, seq_len)
        with amp.autocast(device_type="cuda"):
            loss_fn = model.forward(X, targets=y)
            optimizer.zero_grad()
            scaler.scale(loss_fn).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            print(loss_fn.item())
               