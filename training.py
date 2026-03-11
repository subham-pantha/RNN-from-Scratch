from rnn import RNN
import matplotlib.pyplot as plt
import numpy as np

# data preparation
data = np.sin(np.linspace(0, 50, 1000))
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

rnn = RNN(input_size=1, hidden_size=64, output_size=1, learning_rate=0.001)
seq_length = 20
epochs = 300
loss_history = []

# --- Training Loop ---
print("Training...")
for epoch in range(epochs):
    epoch_loss = 0
    # Sliding window through training data
    for i in range(len(train_data) - seq_length - 1):
        inputs = train_data[i : i + seq_length]
        targets = train_data[i + 1 : i + seq_length + 1]
        
        h, y_preds = rnn.forward_prop(inputs)
        
        # Calculate Mean Squared Error
        step_loss = sum(0.5 * (y_preds[t] - targets[t])**2 for t in range(seq_length))
        epoch_loss += step_loss[0][0]
        
        rnn.backward_prop(inputs, targets, h, y_preds)
    
    loss_history.append(epoch_loss / len(train_data))
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Avg Loss: {loss_history[-1]:.6f}")


# --- Autoregressive Forecasting ---
# We use the model's own predictions as future inputs
seed_seq = list(train_data[-seq_length:])
forecast = []
forecast_steps = 150

for _ in range(forecast_steps):
    # Take the most recent 'seq_length' window
    current_window = seed_seq[-seq_length:]
    _, y_preds = rnn.forward_prop(current_window)
    
    # Get the predicted next value
    next_val = y_preds[seq_length-1][0][0]
    forecast.append(next_val)
    
    # Update the window with the new prediction
    seed_seq.append(next_val)


plt.figure(figsize=(14, 6))

# Subplot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='teal', linewidth=2)
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.grid(True, alpha=0.3)

# Subplot 2: Forecasting
plt.subplot(1, 2, 2)
# Real data for context
plt.plot(range(100), test_data[:100], label="Ground Truth", color='black', alpha=0.3)
# The model's independent forecast
plt.plot(range(100), forecast[:100], label="RNN Forecast", color='red', linestyle='--')
plt.title("Recursive Future Forecasting (100 steps)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()