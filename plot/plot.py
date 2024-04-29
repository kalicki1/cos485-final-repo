import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_table('../results/loss_train.txt', sep='\s+', header=1, names=['epochs', 'd_clean_loss', 'd_noisy_loss', 'g_loss', 'g_conditional_loss', 'timestamp'])

print(df)

# Plotting the lines with different colors
plt.plot(df['epochs'], df['d_clean_loss'],      color='blue', label='Discriminator Clean loss')
plt.plot(df['epochs'], df['d_noisy_loss'],      color='purple', label='Discriminator Noisy loss')
plt.plot(df['epochs'], df['g_loss'],            color='orange', label='Generator loss')
plt.plot(df['epochs'], df['g_conditional_loss'],color='red', label='Generator cond loss')

# Customize the plot
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.title('SEGAN Training loss per Epoch (Batch=50)')
plt.legend()

# Show the plot
plt.show()