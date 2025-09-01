# set_seed(42)
# Example usage:
train_features = 1  # Update with your actual number of input features
autoencoder = Autoencoder(train_features)

device = 'cpu'
autoencoder = autoencoder.to(device)

# Assuming your model is defined as `autoencoder` and data is loaded into `dataloader`
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)  # Adding L2 regularization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30, verbose=True)


def train(accumulate_steps=1):  # Perform update every `accumulate_steps` batches
    autoencoder.train()
    cum_loss = 0
    optimizer.zero_grad()  # Reset gradients at the beginning
    
    for i, data in enumerate(dataloader):
        # Forward pass
        x_recon = autoencoder(data.x, data.edge_index, data.batch)
        
        # Compute loss
        loss = F.mse_loss(data.x, x_recon.view_as(data.x))
        loss.backward()  # Accumulate gradients
        cum_loss += loss.item()
        
        # Update model parameters every `accumulate_steps` batches
        if (i + 1) % accumulate_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()  # Update parameters
            optimizer.zero_grad()  # Reset gradients after update
    
    return cum_loss / len(dataloader)

def validate():
    autoencoder.eval()
    cum_loss = 0
    num_batches = len(val_dataloader)

    # Check if the validation dataloader is empty
    if num_batches == 0:
        print("Validation dataloader is empty. Skipping validation.")
        return float('inf')  # or another appropriate value indicating invalid validation

    with torch.no_grad():
        for data in val_dataloader:
            data = data
            x_recon = autoencoder(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(data.x, x_recon.view_as(data.x))  # Ensure output shape matches input
            cum_loss += loss.item()
    
    return cum_loss / num_batches

# Training loop
for epoch in range(1, 200):
    train_loss = train()
    val_loss = validate()
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    if epoch == 175:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.2  # Reduce learning rate by a factor of 10
            print(f"Manually adjusted learning rate at epoch {epoch}: {param_group['lr']}")
    

    print(f"Epoch {epoch}, Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")
