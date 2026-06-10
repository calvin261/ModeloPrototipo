import re
import sys

path = r'C:\Users\jeffe\Documents\Proyectos\ModeloPrototipo\debug_ui.xml'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

nodes = re.findall(r'<node[^/]*?(?:/>|>)', content)
for n in nodes:
    text = re.search(r'text="([^"]*)"', n)
    desc = re.search(r'content-desc="([^"]*)"', n)
    bounds = re.search(r'bounds="([^"]*)"', n)
    clickable = re.search(r'clickable="(true|false)"', n)
    t = text.group(1) if text else ''
    d = desc.group(1) if desc else ''
    b = bounds.group(1) if bounds else ''
    c = clickable.group(1) if clickable else ''
    if t or d or c == 'true':
        print('text=[%s] desc=[%s] bounds=[%s] clickable=[%s]' % (t, d, b, c))
