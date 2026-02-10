import re

def sanitize_feature_names(names):
    used = set()
    orig2safe, safe2orig = {}, {}
    safe_list = []
    for n in names:
        s = re.sub(r'[^A-Za-z0-9_]', '_', str(n))
        s = re.sub(r'_+', '_', s).strip('_')
        if not s:
            s = 'f'
        base = s
        i = 1
        while s in used:
            i += 1
            s = f"{base}__{i}"
        used.add(s)
        orig2safe[n] = s
        safe2orig[s] = n
        safe_list.append(s)
    return safe_list, orig2safe, safe2orig
