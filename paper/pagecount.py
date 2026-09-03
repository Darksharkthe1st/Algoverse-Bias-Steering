import re, sys
from pathlib import Path

t = Path(sys.argv[1]).read_text(encoding='utf-8')
body = t.split(r'\maketitle', 1)[1].split(r'\begin{thebibliography}')[0]
body = re.sub(r'%.*$', '', body, flags=re.M)


def wc(s):
    s = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', '', s, flags=re.S)
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]+", s))


parts = re.split(r'(?m)^\\(?:sub)?section\*?\{([^}]*)\}', body)
print('%-52s %6s %7s' % ('section', 'words', 'pages'))
print('-' * 68)
tot = wc(parts[0])
print('%-52s %6d %7.2f' % ('(abstract + intro lead)', tot, tot / 575))
for i in range(1, len(parts), 2):
    n = wc(parts[i + 1]); tot += n
    print('%-52s %6d %7.2f' % (parts[i][:50], n, n / 575))

ntab = len(re.findall(r'\\begin\{table\}', body))
tabrows = len(re.findall(r'(?m)&.*\\\\', body))
nfig = len(re.findall(r'\\begin\{figure\}', body))
print('-' * 68)
print('%-52s %6d %7.2f' % ('TOTAL text', tot, tot / 575))
print('%-52s %6d %7.2f' % ('tables (%d envs, %d rows)' % (ntab, tabrows), ntab, ntab * 0.22))
print('%-52s %6d %7.2f' % ('figures (full-width, incl. caption)', nfig, nfig * 0.35))
est = tot / 575 + ntab * 0.22 + nfig * 0.35
print()
print('ESTIMATED TOTAL : %.2f pages     LIMIT = 5     OVER BY %+.2f' % (est, est - 5))
print()
if est > 5:
    print('To fit: cut ~%d words, or drop %d table(s), or drop a figure.'
          % (int((est - 5) * 575), max(1, round((est - 5) / 0.22))))
else:
    print('Fits. Still compile and check the real PDF before trusting this.')
print()
print('Heuristic: 575 words/page, 0.22 page/table, 0.35 page/figure.')
print('Appendices and references are excluded and do not count.')
