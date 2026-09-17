import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f = r"\frac{\partial H}{\partial t}+\frac{a^2}{gA}\frac{\partial Q}{\partial x}=0,\qquad\frac{\partial Q}{\partial t}+gA\frac{\partial H}{\partial x}+\frac{f}{2DA}Q|Q|=0\n"
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('off')
ax.text(.5,.5, f"${f}$", ha='center')
fig.savefig('mathtext_check.png', dpi=200)
print('ok')
