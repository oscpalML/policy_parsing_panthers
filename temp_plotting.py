import matplotlib.pyplot as plt

pow_backt = {
    'esga': '0.94%',
    'gb': '0.93%',
    'dk': '0.88%',
    'fi': '0.83%',
    'hr': '0.83%',
    'pl': '0.90%',
    'si': '0.85%',
    'nl': '0.82%',
    'be': '0.80%',
    'at': '0.91%',
    'es': '0.90%',
    'it': '0.93%',
    'bg': '0.86%',
    'rs': '0.91%',
    'espv': '0.88%',
    'pt': '0.89%',
    'cz': '0.82%',
    'gr': '0.92%',
    'ba': '0.87%',
    'ua': '0.89%',
    'tr': '0.95%',
    'fr': '0.88%',
    'hu': '0.94%',
    'esct': '0.84%',
    'lv': '0.83%',
}

pow_base = {
    'esga': '0.93%',
    'gb': '0.93%',
    'dk': '0.88%',
    'fi': '0.82%',
    'hr': '0.82%',
    'pl': '0.90%',
    'si': '0.86%',
    'nl': '0.81%',
    'be': '0.79%',
    'at': '0.91%',
    'es': '0.90%',
    'it': '0.93%',
    'bg': '0.86%',
    'rs': '0.91%',
    'espv': '0.86%',
    'pt': '0.88%',
    'cz': '0.81%',
    'gr': '0.91%',
    'ba': '0.86%',
    'ua': '0.89%',
    'tr': '0.95%',
    'fr': '0.87%',
    'hu': '0.93%',
    'esct': '0.83%',
    'lv': '0.84%',
}

codes = list(pow_base.keys())
codes.reverse()
diffs = []
for code in codes:
    base_score = float(pow_base[code].replace("%", ""))
    backt_score = float(pow_backt[code].replace("%", ""))
    diffs.append(backt_score-base_score)

plt.bar(codes, diffs)

ax = plt.gca()

# Draw X-axis line
ax.axhline(0, color='black', linewidth=0.5)

ax.axvline(x=13.5, color='red', linewidth=1)

plt.title("Increase in macro-average F1-score from backtranslation")
plt.ylabel("Macro-average F1-score delta")

plt.show()




