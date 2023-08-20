import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to generate and plot sample means
def plot_clt(sample_size, num_samples, distribution):
    plt.figure(figsize=(8, 6))

    means = []
    for _ in range(num_samples):
        sample = distribution(size=sample_size)
        sample_mean = np.mean(sample)
        means.append(sample_mean)

    plt.hist(means, bins=30, density=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of Sample Means (n={sample_size}, {num_samples} samples)')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')

    # Plot the theoretical normal distribution
    sample_mean_mean = np.mean(means)
    sample_mean_std = np.std(means)
    x = np.linspace(np.min(means), np.max(means), 100)
    normal_dist = np.exp(-(x - sample_mean_mean)**2 / (2 * sample_mean_std**2)) / (np.sqrt(2 * np.pi) * sample_mean_std)
    plt.plot(x, normal_dist, color='red', linewidth=2, label='Normal Distribution')
    plt.legend()

    st.pyplot(plt)

# Streamlit UI
st.title('Central Limit Theorem Visualization')
st.sidebar.header('Settings')

# Slider for choosing sample size and number of samples
sample_size = st.sidebar.slider('Sample Size', min_value=5, max_value=1000, value=30, step=5)
num_samples = st.sidebar.slider('Number of Samples', min_value=10, max_value=1000, value=100, step=10)

# Dropdown for choosing distribution type
distribution_type = st.sidebar.selectbox('Select Distribution', ('Uniform', 'Exponential', 'Normal'))
if distribution_type == 'Uniform':
    distribution = np.random.uniform
elif distribution_type == 'Exponential':
    distribution = np.random.exponential
elif distribution_type == 'Normal':
    distribution = np.random.normal

# Explain the Central Limit Theorem
st.write(
    "The Central Limit Theorem (CLT) states that regardless of the distribution of the original "
    "population, the distribution of sample means will tend to follow a normal distribution as the "
    "sample size increases. This principle is fundamental in statistics and has wide-ranging "
    "applications."
)

# Call the function to plot the CLT visualization
st.markdown("### Distribution of Sample Means")
st.write(
    "The plot below shows the distribution of sample means generated from the chosen distribution. "
    "As you increase the sample size and the number of samples, you'll notice that the distribution "
    "tends to become more and more like a normal distribution."
)
plot_clt(sample_size, num_samples, distribution)

# Add a separator and your name
st.sidebar.markdown('---')
st.sidebar.markdown('Created by Harkishan')
