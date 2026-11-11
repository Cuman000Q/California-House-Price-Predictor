from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Alright, first we are settling the data for the fetch_calfornia_housing

data=fetch_california_housing()

# 1 We are assigning the X and Y values that the model will be trained on
X,Y=fetch_california_housing(return_X_y=True)

# 2 Split the data 80% on training and other 20% on the test
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=42)

# 3 We have splitted the data, now we need to make a measurement so during the training it calculates
# The statictics

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# 4 Now we are turning them into the tensors

X_train_tensor=torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor=torch.tensor(X_test_scaled, dtype= torch.float32)

Y_train_tensor= torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
Y_test_tensor= torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

# 5 We need to pack them in the training and test datasets
train_dataset=TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset=TensorDataset(X_test_tensor, Y_test_tensor)

# 6 Load the datasets
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False)

# We have completed with the training and testing datasets,
# 7 We shall start with the AI brain to make this job for us

class Price_predictor(nn.Module):
    def __init__(self, input_size, num_out_neurons=1): # input size is the number of neurons on the matrix has 1D
        super(Price_predictor, self).__init__() # VERY IMPORTANT THING

        self.neural_network=nn.Sequential(
            nn.Linear(input_size, 30), # the neurons of input_size gives signal to 30 neurons in layer1
            nn.ReLU(), # Activation function after the neurons in layer1 were processed
            nn.Linear(30, 15), # 30 neurons give signals to the 15 neurons of layer2
            nn.ReLU(),
            nn.Linear(15, num_out_neurons) # since we need to make the prediction ther is only one output neuron
        )

    def forward(self, x):
        return self.neural_network(x)

predictor_model=Price_predictor(input_size=8, num_out_neurons=1)
# 8 Make the Training loop for the AI model

criterion=nn.MSELoss()
optimizer=optim.Adam(predictor_model.parameters(), lr=0.01) # set the learning rate for this model
# to update the weights

epochs=30

for epoch in range(1, epochs+1):
    predictor_model.train() # specific command to train under this time period
    running_loss=0.0

    for batch_X, batch_Y in test_loader:
        optimizer.zero_grad() # reset old gradients
        output=predictor_model(batch_X) # forward pass the output shape
        loss=criterion(output, batch_Y) # calculating the loss
        loss.backward() # backpropagation of the calculated error 
        optimizer.step() # update the weights
        running_loss+=loss.item()

    if epoch%5==0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {avg_loss:.4f}")

# 9 the final step the report
predictor_model.eval()
all_targets=[]
all_preds=[]

with torch.no_grad():
     for batch_X, batch_Y in test_loader:
           preds = predictor_model(batch_X)
           all_preds.extend(preds.cpu().numpy())
           all_targets.extend(batch_Y.cpu().numpy())

# Calculate Regression Metrics
mse = mean_squared_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print("\n" + "=" * 40)
print(f"=== MEAN SQUARED ERROR (MSE): {mse:.4f} ===")
print(f"=== R² SCORE:                 {r2 * 100:.2f}% ===")
print("=" * 40)
           

