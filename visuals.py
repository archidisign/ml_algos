import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# If in Jupyter Noteboo, add this line to have the plots shown
%matplotlib inline

# ----------------------------
# Simple plot
plt.title("My Plot")
plt.xlabel('Numbers 1')
plt.ylabel('Numbers 2')
'''
Here 
First arg: x values
Second arg: y values
Third arg: desigm, r means red, o means circle
'''
plt.plot([1,2,3,4], [1,4,9,16], 'ro') 
# changes the default axis length
plt.axis([0, 6, 0, 20]) 
plt.show()

# -----------------------------
# Plot multiple lines
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--')
plt.plot(t, t**2, 'bs')
plt.plot(t, t**3, 'g^')
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Curves')
plt.legend(['t', 't**2', 't**3'])
plt.show()

# -----------------------------
# Plot multiple subplots one after each other
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

plt.show()

# -----------------------------
# Set Up Temp Dataset
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]

# Create plots
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c + d, 'k', label='Total message length')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')
plt.show()
