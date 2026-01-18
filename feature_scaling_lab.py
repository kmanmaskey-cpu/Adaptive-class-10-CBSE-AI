import numpy as np
import matplotlib.pyplot as plt
import copy, math

# ====== DEFINE MISSING FUNCTIONS ======
# Since lab_utils_multi doesn't exist, we'll create our own versions

def load_house_data():
    """Load housing dataset"""
    X_train = np.array([[2104, 5, 1, 45], 
                       [1416, 3, 2, 40], 
                       [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
    return X_train, y_train

def zscore_normalize_features(X):
    """Normalize features to have mean=0 and std=1"""
    mu = np.mean(X, axis=0)                 
    sigma = np.std(X, axis=0)                  
    X_norm = (X - mu) / sigma
    
    return (X_norm, mu, sigma)

def plot_cost_i_w(X_train, y_train, w_final, b_final):
    """Plot cost vs iterations"""
    # We'll use a simplified version - you can implement the full one if needed
    print("plot_cost_i_w: This function needs full implementation")
    return

def norm_plot(ax, data):
    """Create normalized distribution plot"""
    # Simplified version
    ax.hist(data, bins=20, density=True, alpha=0.7)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    return ax

def plt_equal_scale(X_train_norm, X_train):
    """Plot comparing scaled vs unscaled features"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Before scaling
    ax[0].scatter(X_train[:, 0], X_train[:, 1])
    ax[0].set_xlabel('Size (sqft)')
    ax[0].set_ylabel('Bedrooms')
    ax[0].set_title('Before Scaling')
    ax[0].grid(True, alpha=0.3)
    
    # After scaling
    ax[1].scatter(X_train_norm[:, 0], X_train_norm[:, 1])
    ax[1].set_xlabel('Size (normalized)')
    ax[1].set_ylabel('Bedrooms (normalized)')
    ax[1].set_title('After Scaling')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def run_gradient_descent(X, y, w_in, b_in, alpha, num_iters, feature_scaling=True):
    """Run gradient descent with optional feature scaling"""
    
    if feature_scaling:
        X_norm, mu, sigma = zscore_normalize_features(X)
        print(f"\nNormalized features. μ={mu}, σ={sigma}")
        X_used = X_norm
    else:
        X_used = X
    
    # Gradient descent implementation
    w = copy.deepcopy(w_in)
    b = b_in
    m, n = X_used.shape
    
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        # Compute gradient
        dj_dw = np.zeros(n)
        dj_db = 0.
        
        for i in range(m):
            error = (np.dot(X_used[i], w) + b) - y[i]
            for j in range(n):
                dj_dw[j] += error * X_used[i, j]
            dj_db += error
            
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        
        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save history for plotting
        if i < 100000:
            # Compute cost
            cost = 0.
            for i in range(m):
                f_wb = np.dot(X_used[i], w) + b
                cost += (f_wb - y[i])**2
            cost = cost / (2 * m)
            J_history.append(cost)
            w_history.append(w.copy())
    
    # Convert weights back if using normalized features
    if feature_scaling:
        w_original = w / sigma
        b_original = b - np.dot(mu / sigma, w)
        return w_original, b_original, J_history, w_history
    
    return w, b, J_history, w_history

# Color dictionary (dlc)
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', 
           dlmagenta='#FF40FF', dlpurple='#7030A0', dldarkblue='#0D5BDC')

# ====== MAIN CODE ======
np.set_printoptions(precision=2)

# Try to load style, but continue if not found
try:
    plt.style.use('./deeplearning.mplstyle')
    print("Using custom style")
except:
    plt.style.use('default')
    print("Using default matplotlib style")

# Load data
print("Loading housing data...")
X_train, y_train = load_house_data()
print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")

# ====== VISUALIZE DATA ======
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.suptitle("Original Features vs Price")
plt.tight_layout()
plt.show()

# ====== FEATURE SCALING DEMONSTRATION ======
print("\n" + "="*60)
print("FEATURE SCALING DEMONSTRATION")
print("="*60)

# Show feature statistics
print("\nOriginal feature statistics:")
print(f"Size:      mean={np.mean(X_train[:,0]):.1f}, std={np.std(X_train[:,0]):.1f}")
print(f"Bedrooms:  mean={np.mean(X_train[:,1]):.1f}, std={np.std(X_train[:,1]):.1f}")
print(f"Floors:    mean={np.mean(X_train[:,2]):.1f}, std={np.std(X_train[:,2]):.1f}")
print(f"Age:       mean={np.mean(X_train[:,3]):.1f}, std={np.std(X_train[:,3]):.1f}")

# Normalize features
X_norm, mu, sigma = zscore_normalize_features(X_train)
print(f"\nNormalized feature statistics:")
print(f"All features now have mean≈0, std≈1")

# Show scaling effect
fig, axes = plt_equal_scale(X_norm, X_train)
plt.show()

# ====== GRADIENT DESCENT WITH DIFFERENT SETTINGS ======
print("\n" + "="*60)
print("GRADIENT DESCENT: SCALED vs UNSCALED")
print("="*60)

# Initial parameters
initial_w = np.zeros(X_train.shape[1])
initial_b = 0.

# Test 1: UNSCALED with tiny alpha
print("\n1. UNSCALED features (α=5e-7, iterations=10000):")
w1, b1, J1, w_hist1 = run_gradient_descent(
    X_train, y_train, initial_w, initial_b, 
    alpha=5e-7, num_iters=10000, feature_scaling=False
)
print(f"   Final cost: {J1[-1]:.2f}")
print(f"   Final w: {w1}")

# Test 2: SCALED with normal alpha
print("\n2. SCALED features (α=0.1, iterations=1000):")
w2, b2, J2, w_hist2 = run_gradient_descent(
    X_train, y_train, initial_w, initial_b, 
    alpha=0.1, num_iters=1000, feature_scaling=True
)
print(f"   Final cost: {J2[-1]:.2f}")
print(f"   Final w: {w2}")

# ====== VISUALIZE LEARNING ======
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Cost vs iterations
axes[0, 0].plot(J1, color=dlc['dlblue'], label='Unscaled (α=5e-7)')
axes[0, 0].plot(J2, color=dlc['dlorange'], label='Scaled (α=0.1)')
axes[0, 0].set_title("Cost vs Iterations")
axes[0, 0].set_xlabel("Iteration")
axes[0, 0].set_ylabel("Cost")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Cost on log scale
axes[0, 1].semilogy(J1, color=dlc['dlblue'], label='Unscaled')
axes[0, 1].semilogy(J2, color=dlc['dlorange'], label='Scaled')
axes[0, 1].set_title("Cost (Log Scale)")
axes[0, 1].set_xlabel("Iteration")
axes[0, 1].set_ylabel("Cost (log)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Normalized distribution of size feature
norm_plot(axes[1, 0], X_train[:, 0])
axes[1, 0].set_title("Size Distribution (Original)")

norm_plot(axes[1, 1], X_norm[:, 0])
axes[1, 1].set_title("Size Distribution (Normalized)")

plt.tight_layout()
plt.show()

# ====== PREDICTIONS ======
print("\n" + "="*60)
print("PREDICTION COMPARISON")
print("="*60)

def predict_house(features, w, b):
    """Predict house price"""
    return np.dot(features, w) + b

# Test on training data
print("\nPredictions on training data:")
for i in range(len(X_train)):
    pred1 = predict_house(X_train[i], w1, b1)  # Unscaled model
    pred2 = predict_house(X_train[i], w2, b2)  # Scaled model
    actual = y_train[i]
    
    print(f"\nHouse {i+1} ({X_train[i,0]} sqft, {X_train[i,1]} bedrooms):")
    print(f"  Actual: ${actual}k")
    print(f"  Unscaled model: ${pred1:.1f}k (error: ${pred1-actual:.1f}k)")
    print(f"  Scaled model: ${pred2:.1f}k (error: ${pred2-actual:.1f}k)")

# Test on new house
print("\n" + "-"*60)
print("PREDICTING NEW HOUSE:")
new_house = np.array([1200, 3, 1, 30])  # 1200 sqft, 3 bedrooms, 1 floor, 30 years

pred_scaled = predict_house(new_house, w2, b2)
print(f"\nNew house: 1200 sqft, 3 bedrooms, 1 floor, 30 years")
print(f"Predicted price (scaled model): ${pred_scaled:.0f}k")
print(f"That's about ${pred_scaled*1000:,.0f}")