import matplotlib.pyplot as plt

plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(6,2))
formula = r'scale\\_pos\\_weight = \\frac{N_{normais}}{N_{fraudes}} = \\frac{90.044}{275} \\approx 327'
ax.text(0.5,0.5, '$' + formula + '$', fontsize=20, ha='center', va='center')
ax.axis('off')
fig.tight_layout(pad=0.2)
fig.savefig('docs/scale_pos_weight_formula.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)
print('imagem criada: docs/scale_pos_weight_formula.png')
